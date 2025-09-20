
import os
import glob

def fix_streamlit_width_issues():
    """Fix all use_container_width=True issues in Python files"""

    # Find all Python files in the project
    python_files = []
    python_files.extend(glob.glob("*.py"))
    python_files.extend(glob.glob("pages/*.py"))
    python_files.extend(glob.glob("pages/**/*.py", recursive=True))

    replacements = [
        ('use_container_width=True', 'use_container_width=True'),
        ("use_container_width=True", 'use_container_width=True'),
        ('width=700', 'width=700'),
        ("width=700", 'width=700')
    ]

    for file_path in python_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                original_content = content

                # Apply all replacements
                for old, new in replacements:
                    content = content.replace(old, new)

                # Only write if content changed
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"✅ Fixed: {file_path}")

            except Exception as e:
                print(f"❌ Error fixing {file_path}: {e}")

if __name__ == "__main__":
    print("🔧 Fixing Streamlit width parameter issues...")
    fix_streamlit_width_issues()
    print("✅ Done! All use_container_width=True replaced with use_container_width=True")
