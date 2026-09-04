"""Test package marker.

Present so pytest puts `deploy/rt595` (not `deploy/rt595/lane/tests`) on
sys.path, which is what makes `import lane` work from a bare `pytest` run at the
repo root.
"""
