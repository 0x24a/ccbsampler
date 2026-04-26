from rich import print
from rich.progress import Progress
import httpx
import typer
import zipfile
import io

HIFIGAN_URL = "https://pub-f707c07a009b432aab05361d2bc09e9e.r2.dev/hifigan.zip"
HNSEP_URL = "https://pub-f707c07a009b432aab05361d2bc09e9e.r2.dev/hnsep.zip"


def _download_file(url: str) -> bytes:
    with Progress(transient=True) as progress:
        file_size = httpx.head(url, follow_redirects=True)
        task = progress.add_task(
            "Downloading", total=int(file_size.headers.get("Content-Length"))
        )
        data = b""
        with httpx.stream("GET", url, follow_redirects=True) as stream:
            for chunk in stream.iter_bytes():
                progress.update(task, advance=len(chunk))
                data += chunk
    return data


app = typer.Typer()


@app.command("models")
def download_models():
    print("[bold]CCBSampler Setup[/bold]")
    print()
    print("[bold]Model Setup[/bold]")
    print("(1/2) [cyan]pc_nsf_hifigan_44.1k_hop512_128bin_2025.02[/cyan]")
    data = _download_file(HIFIGAN_URL)
    print("      Unzipping")
    file = io.BytesIO(data)
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall()
    print("      [green][bold]OK[/bold][/green]")
    print()
    print("(2/2) [cyan]hnsep[/cyan]")
    data = _download_file(HNSEP_URL)
    print("      Unzipping")
    file = io.BytesIO(data)
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extract("model.pt", "hnsep/vr/")
    print("      [green][bold]OK[/bold][/green]")
    print()
    print("[green][bold]All models downloaded![/bold][/green]")


if __name__ == "__main__":
    app()
