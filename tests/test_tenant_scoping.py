from memori import api


class FakeCursor:
    def __init__(self, documents):
        self.documents = documents

    def sort(self, *_args, **_kwargs):
        return self

    def __iter__(self):
        return iter(self.documents)


class FakeCollection:
    def __init__(self):
        self.filters = []

    def find(self, query, projection):
        self.filters.append(query)
        # Mimic MongoDB's tenant filter: return only matching user's documents.
        documents = [
            {"content": "tenant A secret", "type": "fact", "temporal": {"is_current": True}}
            for _ in [0]
            if query.get("user_id") == "tenant-a"
        ]
        return FakeCursor(documents)


class FakeDB:
    def __init__(self, collection):
        self.collection = collection

    def __getitem__(self, name):
        assert name == "memories"
        return self.collection


def test_current_memory_snapshot_filters_by_exact_tenant(monkeypatch):
    from memori import mongodb_connection

    collection = FakeCollection()
    monkeypatch.setattr(mongodb_connection, "db", FakeDB(collection))
    monkeypatch.setattr(api, "_USER_MEMORY_CACHE", {})
    tenant_a = api._get_current_user_memories("tenant-a")
    tenant_b = api._get_current_user_memories("tenant-b")

    assert tenant_a[0]["content"] == "tenant A secret"
    assert tenant_b == []
    assert [item["user_id"] for item in collection.filters] == ["tenant-a", "tenant-b"]
