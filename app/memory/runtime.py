"""Optional long-term-memory orchestration, independent of the Agent graph."""

from app.memory.extractor import MemoryFactExtractor
from app.memory.filtering import filter_candidates
from app.memory.service import MemoryFactService


class LongTermMemoryRuntime:
    def __init__(self, *, fact_service=None, extractor=None):
        self._fact_service = fact_service or MemoryFactService()
        self._extractor = extractor or MemoryFactExtractor()

    def retrieve_for_agent(self, owner_id: str, question: str) -> list[str]:
        return [item.fact_text for item in self._fact_service.search_facts(owner_id, question)]

    def process_successful_user_message(self, owner_id: str, user_message: str) -> None:
        extraction = self._extractor.extract_candidates(user_message)
        accepted, _rejected = filter_candidates(user_message, extraction.facts)
        for candidate in accepted:
            if candidate.memory_text is not None:
                # The model's fact_text is never persisted.
                self._fact_service.save_fact(owner_id, candidate.memory_text,
                                             fact_category=candidate.category)
