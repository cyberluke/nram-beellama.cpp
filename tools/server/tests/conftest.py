import os
from pathlib import Path

import pytest
from utils import *


# ref: https://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test
@pytest.fixture(autouse=True)
def stop_server_after_each_test():
    # do nothing before each test
    yield
    # stop all servers after each test
    instances = set(server_instances)
    for server in instances:
        server.stop()


@pytest.fixture(scope="module", autouse=True)
def do_something(request):
    # Local-model suites validate and provide their own immutable fixture and
    # must not download or launch the unrelated preset inventory.
    if getattr(request.module, "NO_PRELOAD_SERVER_PRESETS", False):
        return
    configured_model = os.environ.get("LLAMA_TEST_MODEL")
    if not configured_model or not Path(configured_model).is_file():
        pytest.skip("native llama-server tests require LLAMA_TEST_MODEL pointing to an available local GGUF model")
    ServerPreset.load_all()
