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
