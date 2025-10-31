from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict
from typing import TYPE_CHECKING
import re


__all__ = [
    "ERGInfo",
    "TestRun",
    "Vehicle",
    "Trailer",
    "Road",
    "Config",
    "OutputQuantities",
    "PlugInfo",
    "SKC",
    "Sensor",
    "Tire",
    "Driver",
    "Traffic",
]

# Regex patterns for numeric value detection
_INTEGER_REGEX = re.compile(r"^[+-]?\d+$")
_NUMERIC_REGEX = re.compile(r"^[+-]?(\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?$")


def is_integer(string: str) -> bool:
    """Check if string represents an integer."""
    return bool(_INTEGER_REGEX.match(string))


def is_numeric(string: str) -> bool:
    """Check if string represents a numeric value."""
    return bool(_NUMERIC_REGEX.match(string))


def is_numeric_array(array: list) -> bool:
    """Check if all elements in array are numeric strings."""
    return all(_NUMERIC_REGEX.match(a) for a in array)


def is_numeric_table(table: list[list]) -> bool:
    """Check if all rows in table contain numeric strings."""
    return all(is_numeric_array(row) for row in table)


def to_numeric_array(array: list) -> list:
    """Convert string array to numeric types where possible."""
    result = array.copy()
    for index, item in enumerate(result):
        if is_integer(item):
            result[index] = int(item)
        elif is_numeric(item):
            result[index] = float(item)
    return result


def to_numeric_table(table: list[list]) -> list:
    """Convert string table to numeric types where possible."""
    return [to_numeric_array(row) for row in table]


def is_nested(array: list) -> bool:
    """Check if array contains nested lists."""
    return isinstance(array, list) and all(isinstance(e, list) for e in array)


@dataclass
class InfoFile(OrderedDict):
    abs_path: Path
    header: str = field(default_factory=str)
    comments: list[str] = field(default_factory=list)
    default_dir: str = field(default_factory=str)

    def __post_init__(self):
        self.abs_path = Path(self.abs_path).resolve()
        if self.abs_path.exists():
            self.read()

    def read(self) -> None:
        reading_table = False

        def flush_table():
            nonlocal reading_table, key, table
            if reading_table:
                self[key] = to_numeric_table(table)
                reading_table = False

        with open(self.abs_path, "r", encoding="utf-8", errors="replace") as f:
            line = f.readline()
            if not line.startswith("#INFOFILE"):
                print(f"{self.abs_path} is not an infofile.")
                return

            self.header = line.strip()
            for line in f:
                if not line.strip():
                    # Empty line
                    pass
                elif line.startswith("#"):
                    # Save comments separately
                    self.comments.append(line)
                elif line.rstrip().endswith(":"):
                    flush_table()
                    # Start of table
                    key, _ = line.split(":", 1)
                    reading_table = True
                    table = []
                elif line.startswith(("\t", "    ")):
                    # Append to table
                    if reading_table:
                        table.append(line.split())
                    else:
                        print(f"Failed to parse line\n->{line}")
                        break
                elif "=" in line:
                    flush_table()
                    # Single line key=value
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = to_numeric_array(value.split())
                    if len(value) == 0:
                        value = None
                    elif len(value) == 1:
                        value = value[0]
                    self[key] = value
                else:
                    print(f"Failed to parse line\n->{line}")
                    break
            flush_table()

    def write(self, file_name: str | None = None) -> None:
        name = file_name if file_name else self.abs_path.name
        with open(name, "w") as f:
            f.writelines(f"{self.header}\n")
            for k, v in self.items():
                if v == None:
                    f.writelines(f"{k} =\n")
                elif is_nested(v):
                    f.writelines(f"{k}:\n")
                    for _v in v:
                        _v = " ".join(map(str, _v)) if isinstance(_v, list) else _v
                        f.writelines(f"\t{_v}\n")
                else:
                    f.writelines(f"{k} = ")
                    v = " ".join(map(str, v)) if isinstance(v, list) else v
                    f.writelines(f"{v}\n")

    def __repr__(self) -> str:
        return self.abs_path.name

    def __str__(self) -> str:
        return self.abs_path.name


class ERGInfo(InfoFile):
    """Parser for ERG info files (.erg.info).

    Contains metadata about ERG binary files including signal names,
    data types, units, and scaling factors.
    """


class TestRun(InfoFile):
    def __init__(self, name: str | Path, project=None) -> None:
        self.default_dir = "Data/TestRun"
        super().__init__(name, project=project)

    def __getitem__(self, key):
        if self.project:
            if key == "Vehicle":
                vehicle = super().__getitem__(key)
                return self.project.vehicle[vehicle]
            elif key == "Trailer":
                trailer = super().__getitem__(key)
                return self.project.trailer[trailer]
            elif key == "Road.FName":
                road = super().__getitem__(key)
                return self.project.road[road]
            else:
                return super().__getitem__(key)
        return super().__getitem__(key)


class Vehicle(InfoFile):
    def __init__(self, name: str | Path, project=None) -> None:
        self.default_dir = "Data/Vehicle"
        super().__init__(name, project=project)


class Trailer(InfoFile):
    def __init__(self, name: str | Path, project=None) -> None:
        self.default_dir = "Data/Trailer"
        super().__init__(name, project=project)


class Road(InfoFile):
    def __init__(self, name: str | Path, project=None) -> None:
        self.default_dir = "Data/Road"
        super().__init__(name, project=project)


class Config(InfoFile):
    def __init__(self, name: str | Path, project=None) -> None:
        self.default_dir = "Data/Config"
        super().__init__(name, project=project)


class OutputQuantities(Config):
    pass


class PlugInfo(InfoFile):
    def __init__(self, name: str | Path, project=None) -> None:
        self.default_dir = "Plugins"
        super().__init__(name, project=project)


class SKC(InfoFile):
    def __init__(self, name: str | Path, project=None) -> None:
        self.default_dir = "Data/Chassis"
        super().__init__(name, project=project)


class Sensor(InfoFile):
    def __init__(self, name: str | Path, project=None) -> None:
        self.default_dir = "Data/Sensor"
        super().__init__(name, project=project)


class Tire(InfoFile):
    def __init__(self, name: str | Path, project=None) -> None:
        self.default_dir = "Data/Tire"
        super().__init__(name, project=project)


class Driver(InfoFile):
    def __init__(self, name: str | Path, project=None) -> None:
        self.default_dir = "Data/Driver"
        super().__init__(name, project=project)


class Traffic(InfoFile):
    def __init__(self, name: str | Path, project=None) -> None:
        self.default_dir = "Data/Traffic"
        super().__init__(name, project=project)
