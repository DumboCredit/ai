from services.bd import vector_store

def get_resume_report(user_id: str):
    results = vector_store.get(
        where={"user_id": user_id},  # filter by user_id tag/metadata
        limit=None  # or a very high number if None is not supported
    )
    return results['documents']