"""Compatibility entry point for CCTV dataset conversion."""
from preprocessing.cctv import convert_cctv_dataset, main, read_json

__all__ = ["convert_cctv_dataset", "main", "read_json"]

if __name__ == "__main__":
    main()
