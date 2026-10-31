from memori.agentic_runtime import (
    _continues_task,
    _is_actionable_task,
    _is_small_talk,
    _requires_artifact,
)


def test_greetings_are_not_actionable_tasks():
    for message in ("Hay", "hey", "Hello!", "How are you?", "Thanks"):
        assert _is_small_talk(message)
        assert not _is_actionable_task(message)


def test_direct_build_request_is_actionable():
    request = "Build a single-page immersive portfolio website."
    assert _is_actionable_task(request)
    assert not _is_small_talk(request)


def test_project_edit_continues_existing_project_task():
    task = {"goal": "Build an immersive portfolio website"}
    assert _continues_task("Change something in the code", task)
    assert _continues_task("Add a contact section to the website", task)


def test_unrelated_new_project_does_not_reuse_existing_task():
    task = {"goal": "Build an immersive portfolio website"}
    assert not _continues_task("Build a separate todo app", task)


def test_legacy_greeting_goal_is_never_continued():
    task = {"goal": "Hay"}
    assert not _continues_task("Change the design", task)

def test_building_a_website_requires_a_saved_artifact():
    assert _requires_artifact(
        "Build a single-page immersive portfolio website as one HTML file."
    )


def test_general_chat_does_not_require_a_saved_artifact():
    assert not _requires_artifact("Hay")
    assert not _requires_artifact("What is the difference between HTML and CSS?")
