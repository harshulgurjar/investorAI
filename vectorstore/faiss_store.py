from pathlib import Path
from types import SimpleNamespace
from langchain_community.vectorstores import FAISS

class FAISSVectorStore:
    def __init__(self,index_path:str ="data/embeddings/faiss_index/index")->None:
        self.index_path=index_path
        self.db =None

    def upload_chunks(self,chunks,embeddings,company:str,year:str,source_file:str)->None:
        for chunk in chunks:
            if not hasattr(chunk,"metadata") or chunk.metadata is None:
                chunk.metadata={}

            chunk.metadata.update({
                "company":company,
                "year":str(year),
                "source_file":source_file
            })
        index_dir=Path(self.index_path)
        if index_dir.exists():
            self.db = FAISS.load_local(index_dir,embeddings,allow_dangerous_deserialization=True)
            self.db.add_documents(chunks)
        else:
            self.db=FAISS.from_documents(chunks,embeddings)

        self.db.save_local(index_dir)
        print(f"Uploaded {len(chunks)} chunks in FAISS.")
    
class Retriever:
    def __init__(self,vector_store:FAISSVectorStore,embeddings=None):
        self.vector_store=vector_store
        self.embeddings=embeddings
        
    def invoke(self,query:str,company:str|None=None,year:int|None=None,top_k:int =20)->list:
        if self.vector_store.db is None:
            self.vector_store.db =FAISS.load_local(
                self.vector_store.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        results =self.vector_store.db.similarity_search(query,k=top_k)

        filtered =[]
        for doc in results:
            meta =doc.metadata or {}
            if company and meta.get("company")!=company:
                continue
            if year and str(meta.get("year")) != str(year):
                continue
            filtered.append(SimpleNamespace(page_content=doc.page_content))
        return filtered

        
        
