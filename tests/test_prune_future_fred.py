import importlib.util
import json
import pathlib

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location("prune_future_fred", SCRIPTS / "prune_future_fred.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeCollection:
    def __init__(self, docs):
        self.docs = dict(docs)
        self.deleted = []

    def get(self, where=None, include=None):
        ids = [i for i, d in self.docs.items() if "observation:" in d]
        return {"ids": ids, "documents": [self.docs[i] for i in ids]}

    def delete(self, ids):
        self.deleted += ids
        for i in ids:
            self.docs.pop(i)


DOCS = {"a": "FEDFUNDS observation: 3.64 on 2026-03-01", "b": "FEDFUNDS observation: 3.64 on 2026-04-01",
        "c": "UNRATE observation: 4.1 on 2026-09-01", "d": "Total net sales 109,417 | 2026-09-01"}


def test_only_fred_chunks_after_period_end_are_selected():
    m = _load()
    assert m.future_fred_ids(["a", "b", "c"], [DOCS["a"], DOCS["b"], DOCS["c"]], "2026-03-28") == ["b", "c"]
    assert m.future_fred_ids(["a"], [DOCS["a"]], "2026-03-01") == []              # same day is kept
    assert m.future_fred_ids(["d"], [DOCS["d"]], "2026-03-28") == []              # not a FRED chunk


def test_dry_run_changes_nothing_and_apply_deletes():
    m = _load()
    coll = FakeCollection(DOCS)
    assert m.prune_collection(coll, "2026-03-28", apply=False) == (3, 2) and coll.deleted == []
    assert m.prune_collection(coll, "2026-03-28", apply=True) == (3, 2)
    assert sorted(coll.deleted) == ["b", "c"] and set(coll.docs) == {"a", "d"}


def test_main_reports_per_scenario(tmp_path, capsys):
    m = _load()
    res = tmp_path / "r.json"
    res.write_text(json.dumps([{"ticker": "AAPL", "period_end": "2026-03-28", "time_period": "AAPL-2026-Q1"}]))
    colls = {"eval_AAPL_2026-03-28": FakeCollection(DOCS)}

    class Client:
        def get_collection(self, name):
            return colls[name]

    m.main(["--results", str(res)], client_factory=lambda: Client())
    out = capsys.readouterr().out
    assert "eval_AAPL_2026-03-28: 3 FRED chunks, 2 dated after 2026-03-28 -> would delete" in out and colls["eval_AAPL_2026-03-28"].docs == DOCS
    m.main(["--results", str(res), "--apply"], client_factory=lambda: Client())
    assert set(colls["eval_AAPL_2026-03-28"].docs) == {"a", "d"}
