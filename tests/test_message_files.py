from memori import message_files


class FakeCursor:
    def __init__(self, docs):
        self.docs = list(docs)

    def sort(self, key, direction):
        self.docs.sort(key=lambda doc: doc.get(key), reverse=direction < 0)
        return self

    def limit(self, count):
        self.docs = self.docs[:count]
        return self

    def __iter__(self):
        return iter(self.docs)


class FakeCollection:
    def __init__(self):
        self.docs = {}

    def create_index(self, *args, **kwargs):
        return "index"

    def update_one(self, query, update, upsert=False):
        key = (query["user_id"], query["session_id"], query["path"])
        doc = self.docs.setdefault(key, dict(update.get("$setOnInsert", {})))
        doc.update(update["$set"])

    def find_one(self, query, projection=None):
        return self.docs.get((query["user_id"], query["session_id"], query["path"]))

    def find(self, query, projection=None):
        docs = [
            doc for doc in self.docs.values()
            if doc.get("user_id") == query.get("user_id")
            and doc.get("session_id") == query.get("session_id")
        ]
        return FakeCursor(docs)

    def delete_one(self, query):
        key = (query["user_id"], query["session_id"], query["path"])
        existed = key in self.docs
        self.docs.pop(key, None)
        return type("DeleteResult", (), {"deleted_count": int(existed)})()


def test_long_message_is_saved_and_read_by_inclusive_line_range(monkeypatch):
    collection = FakeCollection()
    monkeypatch.setattr(message_files, "_collection", lambda: collection)
    saved = message_files.save_agent_message_file(
        "user-a", "session-a", "task-a", "turn-a", "user", "line 1\nline 2\nline 3\n"
    )
    assert saved["path"] == "messages/turn-a-user.txt"
    assert saved["line_count"] == 3
    result = message_files.read_agent_message_file(
        "user-a", "session-a", saved["path"], 1, 2
    )
    assert result["content"] == "line 1\nline 2\n"


def test_message_files_are_isolated_by_user_and_session(monkeypatch):
    collection = FakeCollection()
    monkeypatch.setattr(message_files, "_collection", lambda: collection)
    saved = message_files.save_agent_message_file(
        "user-a", "session-a", "task-a", "turn-a", "assistant", "private"
    )
    import pytest
    with pytest.raises(FileNotFoundError):
        message_files.read_agent_message_file("user-b", "session-a", saved["path"])
    with pytest.raises(FileNotFoundError):
        message_files.read_agent_message_file("user-a", "session-b", saved["path"])


def test_delete_message_file_is_scoped(monkeypatch):
    collection = FakeCollection()
    monkeypatch.setattr(message_files, "_collection", lambda: collection)
    saved = message_files.save_agent_message_file(
        "user-a", "session-a", "task-a", "turn-a", "user", "private"
    )
    assert not message_files.delete_agent_message_file("user-b", "session-a", saved["path"])
    assert message_files.delete_agent_message_file("user-a", "session-a", saved["path"])


def test_message_file_rejects_invalid_line_range(monkeypatch):
    collection = FakeCollection()
    monkeypatch.setattr(message_files, "_collection", lambda: collection)
    saved = message_files.save_agent_message_file(
        "user-a", "session-a", "task-a", "turn-a", "user", "one\ntwo\n"
    )
    import pytest
    with pytest.raises(ValueError, match="1-based"):
        message_files.read_agent_message_file(
            "user-a", "session-a", saved["path"], 0, 1
        )



def test_list_saved_message_paths_is_scoped_and_includes_line_counts(monkeypatch):
    collection = FakeCollection()
    monkeypatch.setattr(message_files, "_collection", lambda: collection)
    message_files.save_agent_message_file(
        "user-a", "session-a", "task-a", "turn-a", "user", "first\nsecond\n"
    )
    message_files.save_agent_message_file(
        "user-b", "session-a", "task-b", "turn-b", "assistant", "private"
    )
    listed = message_files.list_agent_message_files("user-a", "session-a")
    assert len(listed) == 1
    assert listed[0]["path"] == "messages/turn-a-user.txt"
    assert listed[0]["line_count"] == 2
