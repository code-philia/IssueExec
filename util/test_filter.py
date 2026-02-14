# util/test_filter.py
"""
EN

EN blind_spot_analysis、suppletory_retrieval、reranking EN
EN，EN。
"""

from typing import List, Dict, Any


def is_test_file(file_path: str) -> bool:
    """
    EN
    
    EN:
    1. EN tests/ EN /tests/
    2. EN test_ EN _test.py EN
    
    Args:
        file_path: EN
        
    Returns:
        True EN
    """
    if not file_path:
        return False
    
    path_lower = file_path.lower()
    
    # EN1: tests/ EN
    if path_lower.startswith('tests/') or '/tests/' in path_lower:
        return True
    
    # EN2: EN
    # EN
    file_name = path_lower.split('/')[-1] if '/' in path_lower else path_lower
    
    if 'test_' in file_name or file_name.endswith('_test.py'):
        return True
    
    return False


def is_test_location(location: str) -> bool:
    """
    EN
    
    EN: file_path::identifier
    EN identifier EN:
    - ClassName
    - function_name  
    - ClassName.method_name
    
    EN:
    1. EN
    2. EN test_ EN（EN/EN）
    3. EN Test EN（EN）
    
    Args:
        location: EN
        
    Returns:
        True EN
    """
    if not location:
        return False
    
    # EN
    if '::' in location:
        file_path, identifier = location.split('::', 1)
    else:
        # EN
        file_path = location
        identifier = ''
    
    # EN1: EN
    if is_test_file(file_path):
        return True
    
    # EN2 & 3: EN
    if identifier:
        # EN ClassName.method_name EN
        if '.' in identifier:
            class_name, method_name = identifier.split('.', 1)
            # EN
            if class_name.startswith('Test') or method_name.startswith('test_'):
                return True
        else:
            # EN
            if identifier.startswith('Test') or identifier.startswith('test_'):
                return True
    
    return False


def filter_test_locations(locations: List[str]) -> List[str]:
    """
    EN
    
    Args:
        locations: EN
        
    Returns:
        EN
    """
    if not locations:
        return locations
    
    return [loc for loc in locations if not is_test_location(loc)]


def filter_test_files(file_paths: List[str]) -> List[str]:
    """
    EN
    
    Args:
        file_paths: EN
        
    Returns:
        EN
    """
    if not file_paths:
        return file_paths
    
    return [fp for fp in file_paths if not is_test_file(fp)]


def filter_coverage_nodes(coverage_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    EN coverage EN
    
    EN nodes EN，EN edges
    
    Args:
        coverage_dict: coverage EN，EN 'nodes' EN 'edges' EN
        
    Returns:
        EN coverage EN
    """
    if not coverage_dict:
        return coverage_dict
    
    result = dict(coverage_dict)
    
    if 'nodes' in result and result['nodes']:
        result['nodes'] = filter_test_locations(result['nodes'])
    
    return result