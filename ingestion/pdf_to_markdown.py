from pathlib import Path
import pymupdf4llm

class PDFtoMarrkdownConverter:
    """convert PDF to markdown."""
    def convert_pdf(self,pdf_path:str,output_dir:str):
        """
        conert a pdf document to markdown.
        Args:
            pdf_path:source pdf path.
            output_dir:output markdown directory.
        Returns:
            generated markdown file path.
        """
        pdf_file = Path(pdf_path)

        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF not found at {pdf_path}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True,exist_ok=True)
        markdown_content=pymupdf4llm.to_markdown(str(pdf_file))
        markdown_file=output_path/f"{pdf_file.stem}.md"
        markdown_file.write_text(markdown_content,encoding="utf-8")
        return str(markdown_file)

def convert_directory(self,input_dir:str,output_dir:str)->list[str]:
    """
    convert all pdfs from a directory to markdown files.
    Args:
        input_dir:source directory containing pdf files.
        output_dir:output markdown directory.
    Returns:
        list of paths to the generated markdown files.
    """
    input_path = Path(input_dir)
    markdown_files=[]
    for pdf_file in input_path.glob("*.pdf"):
        markdown_file=self.convert_pdf(
            pdf_path=str(pdf_file),
            output_dir=output_dir
        )
        markdown_files.append(markdown_file)
    return markdown_files


if __name__=="__main__":
    repo_root=Path(__file__).resolve().parent[1]
    input_dir = repo_root / "data" / "pdf"
    output_dir = repo_root / "data" / "markdown"    

    converter = PDFtoMarrkdownConverter()
    markdown_files = converter.convert_directory(str(input_dir), str(output_dir))  
    print(f"Successfully converted {len(markdown_files)} PDF files to markdown.")
    for markdown_file in markdown_files:
        print(f"converted:{markdown_file}")
        
