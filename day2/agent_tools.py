import subprocess
from pathlib import Path
from typing import List, Tuple, Optional


class AgentTools:
    """Tools for the coding agent to manipulate the project."""

    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)

    def read_file(self, filepath: str) -> str:
        """Read the contents of a file.

        Args:
            filepath: Path relative to project directory

        Returns:
            File contents as string
        """
        full_path = self.project_path / filepath
        return full_path.read_text()

    def write_file(self, filepath: str, content: str) -> None:
        """Write content to a file, creating directories as needed.

        Args:
            filepath: Path relative to project directory
            content: Content to write
        """
        full_path = self.project_path / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

    def execute_bash_command(self, command: str, cwd: Optional[str] = None) -> Tuple[str, str, int]:
        """Execute a bash command in the project directory.

        Args:
            command: The bash command to execute
            cwd: Working directory relative to project (optional)

        Returns:
            Tuple of (stdout, stderr, returncode)
        """
        work_dir = self.project_path / cwd if cwd else self.project_path

        # Prevent running dev server in Jupyter
        if 'runserver' in command:
            return ("", "Error: Cannot run runserver in this environment", 1)

        result = subprocess.run(
            command,
            shell=True,
            cwd=work_dir,
            capture_output=True,
            text=True
        )

        return (result.stdout, result.stderr, result.returncode)

    def see_file_tree(self, root_dir: str = ".") -> List[str]:
        """List all files in the project.

        Args:
            root_dir: Root directory relative to project (default ".")

        Returns:
            List of relative file paths
        """
        full_path = self.project_path / root_dir
        files = []

        for item in full_path.rglob("*"):
            if item.is_file():
                # Skip __pycache__ and common ignore patterns
                if '__pycache__' not in str(item):
                    files.append(str(item.relative_to(self.project_path)))

        return sorted(files)

    def search_in_files(self, pattern: str, root_dir: str = ".") -> List[Tuple[str, int, str]]:
        """Search for a pattern in all files.

        Args:
            pattern: Text pattern to search for
            root_dir: Root directory relative to project (default ".")

        Returns:
            List of (filepath, line_number, line_content) tuples
        """
        full_path = self.project_path / root_dir
        matches = []

        for file_path in full_path.rglob("*"):
            if file_path.is_file() and '__pycache__' not in str(file_path):
                try:
                    content = file_path.read_text()
                    for line_num, line in enumerate(content.split('\n'), 1):
                        if pattern in line:
                            rel_path = str(file_path.relative_to(self.project_path))
                            matches.append((rel_path, line_num, line))
                except:
                    pass  # Skip binary files

        return matches
