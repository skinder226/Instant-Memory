from memori import message_files


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
