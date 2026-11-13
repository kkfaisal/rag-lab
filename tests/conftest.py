import os
import tempfile

# Point rag-lab at a throwaway project root before it is imported.
os.environ["RAG_LAB_HOME"] = tempfile.mkdtemp(prefix="rag-lab-test-")
