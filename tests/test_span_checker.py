from app.core.span_checker import find_wrong_spans, pick_reference_chunk

SOLAR = {"chunk_id": 1, "text": "About 345 gigawatts of new solar capacity were added in 2023, led by China with 217 gigawatts. "
                                "Solar may become the largest source by the early 2030s."}
POLICY = {"chunk_id": 2, "text": "Countries agreed at COP28 to triple renewable capacity by 2030 and BloombergNEF reported 623 billion dollars."}
CHUNKS = [SOLAR, POLICY]


def test_wrong_number_and_year_are_marked():
    s = "In 2030, about 172 gigawatts of new solar capacity was added globally."
    spans, ref = find_wrong_spans(s, CHUNKS)
    assert ref["chunk_id"] == 1  # about solar, even though 2030 appears in the policy chunk
    assert [sp["text"] for sp in spans] == ["2030", "172"]
    assert all(s[sp["start"]:sp["end"]] == sp["text"] for sp in spans)


def test_correct_sentence_has_no_spans():
    spans, _ = find_wrong_spans("About 345 gigawatts of new solar capacity were added in 2023.", CHUNKS)
    assert spans == []


def test_commas_and_unknown_names():
    chunk = {"chunk_id": 3, "text": "The Three Gorges Dam in China has a capacity of 22,500 megawatts."}
    assert find_wrong_spans("The Three Gorges Dam has 22500 megawatts.", [chunk])[0] == []
    spans, _ = find_wrong_spans("The Itaipu Dam in Brazil has 22,500 megawatts.", [chunk])
    assert {sp["text"] for sp in spans} >= {"Brazil"}


def test_reference_chunk_prefers_topic_overlap():
    assert pick_reference_chunk("Renewable capacity target for 2030", CHUNKS)["chunk_id"] == 2
    assert pick_reference_chunk("anything", []) is None


def test_citations_in_brackets_are_not_numbers():
    spans, _ = find_wrong_spans("This is stated in the context under point [1].", CHUNKS)
    assert spans == []


def test_fix_split_decimals():
    from app.core.query_service import fix_split_decimals
    assert fix_split_decimals("It is 42. 5 percent and magnitude 3. 4.") == "It is 42.5 percent and magnitude 3.4."
    assert fix_split_decimals("It ended. Then 5 people left.") == "It ended. Then 5 people left."
