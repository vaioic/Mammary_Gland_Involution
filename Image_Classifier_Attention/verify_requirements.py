#!/usr/bin/env python3
"""
Verify that requirements.txt matches actual imports in the codebase.
Run this script to check for missing or unused dependencies.
"""

import re
from pathlib import Path
from typing import Set

# Mapping from import names to package names in requirements.txt
IMPORT_TO_PACKAGE = {
    'torch': 'torch',
    'torchvision': 'torchvision',
    'cv2': 'opencv-python',
    'sklearn': 'scikit-learn',
    'openslide': 'openslide-python',
    'h5py': 'h5py',
    'numpy': 'numpy',
    'np': 'numpy',
    'pandas': 'pandas',
    'pd': 'pandas',
    'albumentations': 'albumentations',
    'A': 'albumentations',
    'tqdm': 'tqdm',
    'wandb': 'wandb',
    'matplotlib': 'matplotlib',
    'plt': 'matplotlib',
}

# Built-in modules that don't need to be in requirements.txt
BUILTIN_MODULES = {
    'pathlib', 'typing', 'argparse', 'os', 'sys', 'json',
    'collections', 'itertools', 'functools', 're', 'io',
}

# Local modules (relative imports)
LOCAL_MODULES = {'model', 'dataset', 'train', 'preprocessing'}


def extract_imports_from_file(filepath: Path) -> Set[str]:
    """Extract all import statements from a Python file."""
    imports = set()
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Remove string literals to avoid false positives
    # Simple approach: remove triple-quoted strings
    content = re.sub(r'""".*?"""', '', content, flags=re.DOTALL)
    content = re.sub(r"'''.*?'''", '', content, flags=re.DOTALL)
    
    # Find import statements (only at start of lines)
    for line in content.split('\n'):
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('#'):
            continue
        
        # Match: import module
        match = re.match(r'^import\s+(\w+)', line)
        if match:
            imports.add(match.group(1))
            continue
        
        # Match: from module import ...
        match = re.match(r'^from\s+(\w+)', line)
        if match:
            imports.add(match.group(1))
    
    return imports


def extract_packages_from_requirements(filepath: Path) -> Set[str]:
    """Extract package names from requirements.txt."""
    packages = set()
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Extract package name (before >= or ==)
            package = re.split(r'[><=]', line)[0].strip()
            packages.add(package)
    
    return packages


def main():
    """Main verification function."""
    script_dir = Path(__file__).parent
    
    # Find all Python files
    python_files = list(script_dir.glob('*.py'))
    python_files = [f for f in python_files if f.name != 'verify_requirements.py']
    
    print("=" * 60)
    print("Requirements.txt Verification")
    print("=" * 60)
    
    # Extract all imports from all files
    all_imports = set()
    for pyfile in python_files:
        imports = extract_imports_from_file(pyfile)
        all_imports.update(imports)
        print(f"\n{pyfile.name}: {len(imports)} imports")
    
    # Filter out built-in and local modules
    external_imports = all_imports - BUILTIN_MODULES - LOCAL_MODULES
    
    print(f"\n{'=' * 60}")
    print(f"External packages imported: {len(external_imports)}")
    print(f"{'=' * 60}")
    for imp in sorted(external_imports):
        print(f"  - {imp}")
    
    # Map to package names
    required_packages = set()
    unmapped_imports = set()
    
    for imp in external_imports:
        if imp in IMPORT_TO_PACKAGE:
            required_packages.add(IMPORT_TO_PACKAGE[imp])
        else:
            unmapped_imports.add(imp)
    
    # Read requirements.txt
    req_file = script_dir / 'requirements.txt'
    if req_file.exists():
        listed_packages = extract_packages_from_requirements(req_file)
    else:
        print("\n❌ ERROR: requirements.txt not found!")
        return
    
    # Compare
    missing_in_requirements = required_packages - listed_packages
    unused_in_requirements = listed_packages - required_packages
    
    print(f"\n{'=' * 60}")
    print("Analysis Results")
    print(f"{'=' * 60}")
    
    if not missing_in_requirements and not unused_in_requirements and not unmapped_imports:
        print("✅ All checks passed! Requirements.txt is accurate.")
    else:
        if missing_in_requirements:
            print("\n❌ Missing in requirements.txt:")
            for pkg in sorted(missing_in_requirements):
                print(f"  - {pkg}")
        
        if unused_in_requirements:
            print("\n⚠️  In requirements.txt but not imported:")
            for pkg in sorted(unused_in_requirements):
                print(f"  - {pkg}")
        
        if unmapped_imports:
            print("\n⚠️  Imported but not in mapping (might be built-in or typo):")
            for imp in sorted(unmapped_imports):
                print(f"  - {imp}")
    
    print(f"\n{'=' * 60}")
    print("Required packages in requirements.txt:")
    print(f"{'=' * 60}")
    for pkg in sorted(listed_packages):
        status = "✅" if pkg in required_packages else "⚠️ "
        print(f"{status} {pkg}")


if __name__ == "__main__":
    main()
