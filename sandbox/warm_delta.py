"""Image build step: resolve Delta's jars once, so the sandbox never needs the network."""

import shutil
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

ivy = Path("/tmp/ivy")  # noqa: S108 - image build only
builder = (
    SparkSession.builder.master("local[1]")
    .config("spark.jars.ivy", str(ivy))
    .config("spark.ui.enabled", "false")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.stop()
target = Path("/opt/sandbox/jars")
target.mkdir(parents=True, exist_ok=True)
for jar in (ivy / "jars").glob("*.jar"):
    shutil.copy(jar, target)
print(sorted(p.name for p in target.iterdir()))
