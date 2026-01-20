"""
Test script for Part 2: AgentTools Class

Tests all the AgentTools methods to ensure they work correctly.
Run this from the day2 directory.
"""

import os
import shutil
import tempfile
from pathlib import Path

print("=" * 60)
print("Part 2: Testing AgentTools Class")
print("=" * 60)

# Import AgentTools
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "day2"))
from agent_tools import AgentTools

# Create a temporary test directory
test_dir = Path(tempfile.mkdtemp(prefix="agent_tools_test_"))
print(f"\nCreated test directory: {test_dir}")

# Create a simple test project structure
test_project = test_dir / "test_project"
test_project.mkdir()

# Create some test files
(test_project / "README.md").write_text("# Test Project")
(test_project / "src").mkdir()
(test_project / "src" / "main.py").write_text("""
def home():
    return "Hello, World!"

def greet(name):
    return f"Hello, {name}!"
""")
(test_project / "src" / "__init__.py").write_text("")

print("\n--- Test 1: Initialize AgentTools ---")
tools = AgentTools(test_project)
print(f"[OK] AgentTools initialized with path: {tools.project_path}")
print(f"[OK] project_path stored correctly: {tools.project_path == test_project}")

print("\n--- Test 2: see_file_tree() ---")
files = tools.see_file_tree()
print(f"[OK] Found {len(files)} files:")
for f in files:
    print(f"  - {f}")
assert "README.md" in files
# Handle both forward and backward slashes for cross-platform compatibility
assert any("main.py" in f for f in files)
assert "__pycache__" not in str(files)

print("\n--- Test 3: read_file() ---")
content = tools.read_file("README.md")
print(f"[OK] Read README.md:")
print(f"  Content: {repr(content)}")
assert content == "# Test Project"

content = tools.read_file("src/main.py")
print(f"[OK] Read src/main.py:")
assert "def home():" in content
assert "def greet(name):" in content

print("\n--- Test 4: write_file() - new file ---")
tools.write_file("src/new_file.py", "# New file\nprint('hello')")
new_content = tools.read_file("src/new_file.py")
print(f"[OK] Wrote and read new_file.py:")
print(f"  Content: {repr(new_content)}")
assert new_content == "# New file\nprint('hello')"

print("\n--- Test 5: write_file() - nested directories ---")
tools.write_file("deeply/nested/path/file.txt", "nested content")
nested_content = tools.read_file("deeply/nested/path/file.txt")
print(f"[OK] Created nested directories and wrote file:")
print(f"  Content: {repr(nested_content)}")
assert nested_content == "nested content"

print("\n--- Test 6: search_in_files() ---")
matches = tools.search_in_files("def")
print(f"[OK] Searching for 'def' found {len(matches)} matches:")
for path, line_num, line in matches:
    print(f"  - {path}:{line_num}: {line.strip()}")
assert len(matches) >= 2  # Should find at least home() and greet()

matches = tools.search_in_files("home")
print(f"[OK] Searching for 'home' found {len(matches)} matches:")
for path, line_num, line in matches:
    print(f"  - {path}:{line_num}: {line.strip()}")
assert len(matches) >= 1

print("\n--- Test 7: execute_bash_command() - simple command ---")
stdout, stderr, returncode = tools.execute_bash_command("echo hello")
print(f"[OK] Executed 'echo hello':")
print(f"  stdout: {repr(stdout.strip())}")
print(f"  stderr: {repr(stderr.strip())}")
print(f"  returncode: {returncode}")
assert "hello" in stdout
assert returncode == 0

print("\n--- Test 8: execute_bash_command() - runserver blocked ---")
stdout, stderr, returncode = tools.execute_bash_command("python manage.py runserver")
print(f"[OK] Executed runserver command (should be blocked):")
print(f"  stdout: {repr(stdout)}")
print(f"  stderr: {repr(stderr)}")
print(f"  returncode: {returncode}")
assert returncode == 1
assert "runserver" in stderr or "Error" in stderr

print("\n--- Test 9: execute_bash_command() - with cwd ---")
(test_project / "subdir").mkdir()
(test_project / "subdir" / "test.txt").write_text("subdir content")
stdout, stderr, returncode = tools.execute_bash_command("cat test.txt", cwd="subdir")
print(f"[OK] Executed command with cwd='subdir':")
print(f"  stdout: {repr(stdout.strip())}")
assert "subdir content" in stdout

print("\n--- Test 10: Copy template function (start()) ---")

def start(project_name, template_dir=test_project):
    """Copy template to new project directory."""
    target = test_dir / project_name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(template_dir, target)
    return target

new_project = start("my_new_project")
print(f"[OK] Copied template to: {new_project}")
assert (new_project / "README.md").exists()
assert (new_project / "src" / "main.py").exists()

print("\n" + "=" * 60)
print("All Part 2 tests passed!")
print("=" * 60)

# Cleanup
print(f"\nCleaning up test directory: {test_dir}")
shutil.rmtree(test_dir)
print("[OK] Cleanup complete")
