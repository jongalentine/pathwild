#!/usr/bin/env python3
"""
Replace all references to Area 048 with Wyoming Hunt Area 048 (Leo-Hanna area).
Updates notebooks, scripts, documentation, and data files.
"""

import re
from pathlib import Path
import json

# Area 048 coordinates (Leo-Hanna area center)
AREA_048_LAT = 41.8350  # Approximate center
AREA_048_LON = -106.4250  # Approximate center
AREA_048_NAME = "Area 048"
AREA_048_FULL_NAME = "Wyoming Hunt Area 048 (Leo-Hanna area)"

# Old Area 048 coordinates
area_048_lat = 41.835
area_048_lon = -106.425

def replace_in_file(file_path: Path):
    """Replace Area 048 references with Area 048 in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace coordinates
        content = content.replace(f"{area_048_lat}", f"{AREA_048_LAT}")
        content = content.replace(f"{area_048_lon}", f"{AREA_048_LON}")
        
        # Replace variable names
        content = re.sub(r'area_048_lat', 'area_048_lat', content, flags=re.IGNORECASE)
        content = re.sub(r'area_048_lon', 'area_048_lon', content, flags=re.IGNORECASE)
        content = re.sub(r'area_048_', 'area_048_', content, flags=re.IGNORECASE)
        
        # Replace display names
        content = re.sub(r'Area 048', AREA_048_NAME, content)
        content = re.sub(r'area 048', AREA_048_NAME.lower(), content, flags=re.IGNORECASE)
        
        # Replace distance column names
        content = re.sub(r'distance_to_area_048_km', 'distance_to_area_048_km', content, flags=re.IGNORECASE)
        
        # Replace in comments and descriptions
        content = re.sub(r'Area 048, Wyoming', AREA_048_FULL_NAME, content, flags=re.IGNORECASE)
        content = re.sub(r'Area 048 area', f'{AREA_048_NAME} area', content, flags=re.IGNORECASE)
        content = re.sub(r'near Area 048', f'near {AREA_048_NAME}', content, flags=re.IGNORECASE)
        content = re.sub(r'from Area 048', f'from {AREA_048_NAME}', content, flags=re.IGNORECASE)
        content = re.sub(r'to Area 048', f'to {AREA_048_NAME}', content, flags=re.IGNORECASE)
        
        # Replace file names in paths
        content = re.sub(r'south_bighorn_routes_area_048', 'south_bighorn_routes_area_048', content, flags=re.IGNORECASE)
        content = re.sub(r'national_refuge_points_area_048', 'national_refuge_points_area_048', content, flags=re.IGNORECASE)
        
        # Update specific descriptions
        content = re.sub(r'perfect for Area 048, Wyoming', f'perfect for {AREA_048_FULL_NAME}', content, flags=re.IGNORECASE)
        content = re.sub(r'~200 miles from Area 048', f'~200 miles from {AREA_048_NAME}', content, flags=re.IGNORECASE)
        content = re.sub(r'~200 miles from Area 048', f'~200 miles from {AREA_048_NAME}', content, flags=re.IGNORECASE)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        return False

def replace_in_notebook(file_path: Path):
    """Replace Area 048 references in a Jupyter notebook"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        modified = False
        for cell in notebook.get('cells', []):
            if 'source' in cell:
                if isinstance(cell['source'], list):
                    original = ''.join(cell['source'])
                else:
                    original = cell['source']
                
                # Replace coordinates
                new_content = original.replace(f"{area_048_lat}", f"{AREA_048_LAT}")
                new_content = new_content.replace(f"{area_048_lon}", f"{AREA_048_LON}")
                
                # Replace variable names
                new_content = re.sub(r'area_048_lat', 'area_048_lat', new_content, flags=re.IGNORECASE)
                new_content = re.sub(r'area_048_lon', 'area_048_lon', new_content, flags=re.IGNORECASE)
                new_content = re.sub(r'area_048_', 'area_048_', new_content, flags=re.IGNORECASE)
                
                # Replace display names
                new_content = re.sub(r'Area 048', AREA_048_NAME, new_content)
                new_content = re.sub(r'area 048', AREA_048_NAME.lower(), new_content, flags=re.IGNORECASE)
                
                # Replace distance column names
                new_content = re.sub(r'distance_to_area_048_km', 'distance_to_area_048_km', new_content, flags=re.IGNORECASE)
                
                # Replace descriptions
                new_content = re.sub(r'Area 048, Wyoming', AREA_048_FULL_NAME, new_content, flags=re.IGNORECASE)
                new_content = re.sub(r'Area 048 area', f'{AREA_048_NAME} area', new_content, flags=re.IGNORECASE)
                new_content = re.sub(r'near Area 048', f'near {AREA_048_NAME}', new_content, flags=re.IGNORECASE)
                new_content = re.sub(r'from Area 048', f'from {AREA_048_NAME}', new_content, flags=re.IGNORECASE)
                new_content = re.sub(r'to Area 048', f'to {AREA_048_NAME}', new_content, flags=re.IGNORECASE)
                new_content = re.sub(r'perfect for Area 048, Wyoming', f'perfect for {AREA_048_FULL_NAME}', new_content, flags=re.IGNORECASE)
                new_content = re.sub(r'~200 miles from Area 048', f'~200 miles from {AREA_048_NAME}', new_content, flags=re.IGNORECASE)
                new_content = re.sub(r'south_bighorn_routes_area_048', 'south_bighorn_routes_area_048', new_content, flags=re.IGNORECASE)
                
                if new_content != original:
                    if isinstance(cell['source'], list):
                        cell['source'] = [new_content]
                    else:
                        cell['source'] = new_content
                    modified = True
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=2, ensure_ascii=False)
            return True
        return False
    except Exception as e:
        print(f"  Error processing notebook {file_path}: {e}")
        return False

def main():
    """Main replacement function"""
    project_root = Path(".")
    
    print("=" * 70)
    print("REPLACING area 048 WITH AREA 048")
    print("=" * 70)
    print(f"\nArea 048 coordinates: {AREA_048_LAT:.4f}°, {AREA_048_LON:.4f}°")
    print(f"Area 048 name: {AREA_048_FULL_NAME}")
    
    files_modified = []
    
    # Process notebooks
    print("\nProcessing notebooks...")
    for nb_file in project_root.glob("notebooks/**/*.ipynb"):
        if replace_in_notebook(nb_file):
            files_modified.append(nb_file)
            print(f"  ✓ {nb_file}")
    
    # Process Python scripts
    print("\nProcessing Python scripts...")
    for py_file in project_root.glob("scripts/**/*.py"):
        if replace_in_file(py_file):
            files_modified.append(py_file)
            print(f"  ✓ {py_file}")
    
    # Process documentation
    print("\nProcessing documentation...")
    for doc_file in project_root.glob("docs/**/*.md"):
        if replace_in_file(doc_file):
            files_modified.append(doc_file)
            print(f"  ✓ {doc_file}")
    
    # Process data files (CSV headers)
    print("\nProcessing data files...")
    for csv_file in project_root.glob("data/**/*.csv"):
        if replace_in_file(csv_file):
            files_modified.append(csv_file)
            print(f"  ✓ {csv_file}")
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: Modified {len(files_modified)} files")
    print("=" * 70)
    
    if files_modified:
        print("\nFiles modified:")
        for f in files_modified:
            print(f"  - {f}")
    
    print("\n✓ Replacement complete!")
    print(f"\nNote: You may need to regenerate processed CSV files")
    print(f"      with the new 'distance_to_area_048_km' column name.")

if __name__ == "__main__":
    main()
