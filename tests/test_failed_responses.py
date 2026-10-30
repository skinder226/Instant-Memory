from memori.failed_responses import parse_response_files


def test_parse_complete_marked_file():
    response = """Intro
## Create File (index.html)
<!doctype html>
<html></html>
## Ending file (index.html)
Done"""
    complete, partial = parse_response_files(response)
    assert complete == [{"path": "index.html", "content": "<!doctype html>\n<html></html>"}]
    assert partial == []


def test_preserve_unfinished_file_block_after_interruption():
    response = """## Create File (src/app.py)
print('partial')
"""
    complete, partial = parse_response_files(response, allow_incomplete=True)
    assert complete == []
    assert partial == [{"path": "src/app.py", "content": "print('partial')"}]


def test_reject_path_traversal():
    response = """## Create File (../secrets.txt)
do not write
## Ending file (../secrets.txt)
"""
    complete, partial = parse_response_files(response, allow_incomplete=True)
    assert complete == []
    assert partial == []


def test_multiple_files_are_parsed_independently():
    response = """## Create File (index.html)
<html></html>
## Ending file (index.html)
## Create File (assets/app.js)
console.log('ok')
## Ending file (assets/app.js)
"""
    complete, partial = parse_response_files(response)
    assert [item["path"] for item in complete] == ["index.html", "assets/app.js"]
    assert partial == []



def test_save_failed_response_writes_to_collection(monkeypatch):
    from memori import failed_responses

    class FakeCollection:
        def __init__(self):
            self.documents = {}

        def update_one(self, query, update, upsert=False):
            assert upsert is True
            self.documents.setdefault((query["user_id"], query["turn_id"]), update["$setOnInsert"])

        def find_one(self, query, projection=None, **kwargs):
            return self.documents.get((query["user_id"], query["turn_id"]))

    collection = FakeCollection()
    monkeypatch.setattr(failed_responses, "_collection", lambda: collection)

    saved = failed_responses.save_failed_response(
        "user-1", "session-1", "task-1", "turn-1", "partial answer",
        error_type="ProviderError", error_message="token limit",
    )

    assert saved["user_id"] == "user-1"
    assert saved["session_id"] == "session-1"
    assert saved["task_id"] == "task-1"
    assert saved["partial_response"] == "partial answer"
    assert saved["status"] == "failed"
    assert ("user-1", "turn-1") in collection.documents


def test_save_failed_response_redacts_provider_secrets(monkeypatch):
    from memori import failed_responses

    class FakeCollection:
        def __init__(self):
            self.document = None

        def update_one(self, query, update, upsert=False):
            self.document = update["$setOnInsert"]

        def find_one(self, query, projection=None, **kwargs):
            return self.document

    collection = FakeCollection()
    monkeypatch.setattr(failed_responses, "_collection", lambda: collection)
    saved = failed_responses.save_failed_response(
        "user-1", "session-1", "task-1", "turn-2", "",
        error_type="ProviderError",
        error_message="api_key=sk-1234567890abcdef",
    )
    assert "sk-1234567890abcdef" not in saved["error_message"]
    assert "[REDACTED]" in saved["error_message"]
