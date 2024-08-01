from pathlib import Path


class EditorExport:

    def asset_handling(resources_folder: Path, write_path: Path):
        """
        Copys the Editor_AssetHandling.cs file to the unity project
        :return nothing:
        """
        editor_file: Path = resources_folder / "Editor_AssetHandling.cs"
        destination: Path = write_path / "Editor_AssetHandling.cs"
        if not destination.exists():
            destination.write_bytes(editor_file.read_bytes())


EditorExport.asset_handling = staticmethod(EditorExport.asset_handling)

