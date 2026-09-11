import logging
from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path

from .timing import log_duration

logger = logging.getLogger(__name__)


def process_pdf_files(pdf_directory: str):
    pdf_path = Path(pdf_directory)
    pdf_files = list(pdf_path.glob("**/*.pdf"))

    all_documents = []

    logger.info(f"Number of files in directory: {pdf_directory} is {len(pdf_files)}")

    for pdf_file in pdf_files:
        logger.info(f"Processing file: {pdf_file.name}")
        try:
            with log_duration(logger, f"Loading {pdf_file.name}"):
                loader = PyMuPDFLoader(str(pdf_file))
                documents = loader.load()

            for document in documents:
                document.metadata["source_file"] = pdf_file.name
                document.metadata["file_type"] = "pdf"

            all_documents.extend(documents)

            logger.info(f"Loaded {len(documents)} pages for file: {pdf_file}")
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {e}")

    logger.info(f"Total documents loaded: {len(all_documents)}")

    return all_documents
