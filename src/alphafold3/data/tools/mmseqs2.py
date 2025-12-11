"""MMseqs2 MSA tool."""

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Sequence

from alphafold3.data import parsers
from alphafold3.data.tools import msa_tool


def run_with_logging(cmd: Sequence[str], env: dict | None = None) -> subprocess.CompletedProcess:
    """Runs command and logs stdout/stderr."""
    logging.info("Running command: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, capture_output=True, check=True, text=True,
            env=env
        )
        if result.stdout:
            logging.info("stdout:\n%s\n", result.stdout)
        if result.stderr:
            logging.info("stderr:\n%s\n", result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        logging.error(f"stdout: {e.stdout}")
        logging.error(f"stderr: {e.stderr}")
        raise


class MMseqs2(msa_tool.MsaTool):
    """Python wrapper for MMseqs2."""

    def __init__(
        self,
        *,
        binary_path: str,
        database_path: str,
        n_cpu: int = 8,
        e_value: float = 0.0001,
        max_sequences: int = 10000,
        sensitivity: float = 7.5,
        gpu_devices: tuple[str, ...] | None = None,
    ):
        self.gpu_db = None
        """Initializes the MMseqs2 runner.

        Args:
            binary_path: Path to the MMseqs2 binary.
            database_path: Path to the sequence database.
            n_cpu: Number of CPUs to use.
            e_value: E-value cutoff.
            max_sequences: Maximum number of sequences to return.
            sensitivity: Search sensitivity (from -s parameter).
            gpu_devices: List of GPU devices to use (e.g. ["0", "1"]).
        """
        self.binary_path = binary_path
        self.database_path = database_path
        self.n_cpu = n_cpu
        self.e_value = e_value
        self.max_sequences = max_sequences
        self.sensitivity = sensitivity
        self.use_gpu = gpu_devices is not None and len(gpu_devices) > 0
        self.gpu_devices = gpu_devices
        logging.info(f"[DEBUG] MMseqs2 init: gpu_devices={gpu_devices}, use_gpu={self.use_gpu}")
        logging.info(f"[DEBUG] MMseqs2 init: database_path={database_path}")

        if not os.path.exists(self.database_path):
            raise ValueError(f"Database file not found: {self.database_path}")

        db_dir = os.path.dirname(self.database_path)
        db_basename = os.path.basename(self.database_path)
        db_name = os.path.splitext(db_basename)[0]
        
        self.mmseqs_dir = os.path.join(db_dir, f"{db_name}_mmseqs2")
        os.makedirs(self.mmseqs_dir, exist_ok=True)
        
        self.base_db = os.path.join(self.mmseqs_dir, db_name)
        self.base_db_type = self.base_db + ".dbtype"
        self.base_db_source = self.base_db + ".source"
        logging.info(f"[DEBUG] MMseqs2 init: mmseqs_dir={self.mmseqs_dir}")
        logging.info(f"[DEBUG] MMseqs2 init: base_db={self.base_db}")

        if not os.path.exists(self.base_db_type):
            logging.info(f"Creating base MMseqs2 database at {self.base_db}")
            cmd = [
                self.binary_path,
                "createdb",
                self.database_path,
                self.base_db,
                "--dbtype", "1",
                "--compressed", "0"
            ]
            try:
                run_with_logging(cmd)
                if not os.path.exists(self.base_db_source):
                    os.symlink(self.database_path, self.base_db_source)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to create base database: {e}")

        logging.info(f"[DEBUG] Checking GPU setup: use_gpu={self.use_gpu}")
        if self.use_gpu:
            self.gpu_index_dir = os.path.join(self.mmseqs_dir, f"{db_name}_gpu_index")
            os.makedirs(self.gpu_index_dir, exist_ok=True)
            
            self.gpu_db = os.path.join(self.gpu_index_dir, f"{db_name}_gpu")
            logging.info(f"[DEBUG] GPU db path set to: {self.gpu_db}")
            logging.info(f"[DEBUG] GPU index dir: {self.gpu_index_dir}")

            required_exts = [".dbtype", ".index", ".lookup", "_h"]
            existing_files = {}
            for ext in required_exts:
                path = self.gpu_db + ext
                exists = os.path.exists(path)
                existing_files[ext] = exists
                logging.info(f"[DEBUG] GPU index file check: {path} exists={exists}")
            
            if not all(existing_files.values()):
                logging.info(f"Creating GPU-optimized database in {self.gpu_index_dir}")
                
                cmd = [
                    self.binary_path,
                    "makepaddedseqdb",
                    self.base_db,
                    self.gpu_db
                ]
                try:
                    run_with_logging(cmd)
                except subprocess.CalledProcessError as e:
                    logging.error(f"Failed to create GPU-optimized database: {e}")
                    self.use_gpu = False
                    return

                with tempfile.TemporaryDirectory() as tmp_dir:
                    cmd = [
                        self.binary_path,
                        "createindex",
                        self.gpu_db,
                        tmp_dir,
                        "--remove-tmp-files", "1",
                        "--threads", str(self.n_cpu),
                        "--comp-bias-corr", "0",
                        "--search-type", "1",
                        "--mask", "0"
                    ]
                    try:
                        run_with_logging(cmd)
                        logging.info("GPU index creation completed successfully")
                    except subprocess.CalledProcessError as e:
                        logging.error(f"Failed to create GPU index: {e}")
                        self.use_gpu = False
            else:
                logging.info(f"[DEBUG] All GPU index files exist, skipping creation")

    def _gpu_search(
        self, 
        query_path: str, 
        result_m8: str, 
        tmp_dir: str,
        target_sequence: str,
    ) -> msa_tool.MsaToolResult:
        """Hybrid GPU search: GPU prefilter + CPU realignment for A3M generation."""
        gpu_db = self.gpu_db
        logging.info(f"[DEBUG] _gpu_search (hybrid) called: gpu_db={gpu_db}")
        
        if not gpu_db:
            logging.warning("GPU database not available, falling back to CPU search")
            return self._cpu_search(query_path, result_m8, tmp_dir, target_sequence)
        
        query_db = os.path.join(tmp_dir, "query_db")
        gpu_result_db = os.path.join(tmp_dir, "gpu_result")
        
        cmd = [self.binary_path, "createdb", query_path, query_db]
        run_with_logging(cmd)
        
        env = os.environ.copy()
        original_cuda_devices = env.get("CUDA_VISIBLE_DEVICES")
        
        if self.gpu_devices:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(self.gpu_devices)
            logging.info(f"[DEBUG] Setting CUDA_VISIBLE_DEVICES={env[CUDA_VISIBLE_DEVICES]} for MMseqs2 GPU search")
        
        try:
            logging.info("[DEBUG] Step 1: Running GPU search (prefilter)")
            cmd = [
                self.binary_path,
                "search",
                query_db,
                gpu_db,
                gpu_result_db,
                tmp_dir,
                "--threads", str(self.n_cpu),
                "--max-seqs", str(self.max_sequences),
                "-s", str(self.sensitivity),
                "-e", str(self.e_value),
                "--db-load-mode", "0",
                "--comp-bias-corr", "0",
                "--mask", "0",
                "--exact-kmer-matching", "1",
                "--gpu", "1"
            ]
            run_with_logging(cmd, env=env)
            
            logging.info("[DEBUG] Step 2: Realigning against base_db for A3M generation")
            realign_db = os.path.join(tmp_dir, "realign")
            cmd = [
                self.binary_path,
                "align",
                query_db,
                self.base_db,
                gpu_result_db,
                realign_db,
                "--threads", str(self.n_cpu),
                "-e", str(self.e_value),
                "-a",
                "--alignment-mode", "3",
            ]
            run_with_logging(cmd)
            
            cmd = [
                self.binary_path,
                "convertalis",
                query_db,
                self.base_db,
                realign_db,
                result_m8,
                "--format-output", "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits",
            ]
            run_with_logging(cmd)
            
            return self._process_search_results(
                result_db=realign_db,
                target_sequence=target_sequence,
                query_db_path=query_db,
                tmp_dir=tmp_dir,
                target_db=self.base_db
            )
            
        except subprocess.CalledProcessError as e:
            logging.error(f"[DEBUG] GPU hybrid search failed: {e}")
            logging.warning("[DEBUG] Falling back to CPU search")
            return self._cpu_search(query_path, result_m8, tmp_dir, target_sequence)
            
        finally:
            if original_cuda_devices is not None:
                env["CUDA_VISIBLE_DEVICES"] = original_cuda_devices
            elif "CUDA_VISIBLE_DEVICES" in env:
                del env["CUDA_VISIBLE_DEVICES"]

    def _process_search_results(
        self,
        result_db: str,
        target_sequence: str,
        query_db_path: str,
        tmp_dir: str,
        target_db: str = None,
    ) -> msa_tool.MsaToolResult:
        """Processes MMseqs2 search results to A3M format."""
        if not os.path.exists(result_db) or os.path.getsize(result_db) == 0:
            logging.error(f"No search results found in {result_db}")
            return msa_tool.MsaToolResult(a3m="", target_sequence=target_sequence, e_value=self.e_value)

        if target_db is None:
            target_db = self.base_db

        result_a3m_path = os.path.join(tmp_dir, "result.a3m")
        cmd = [
            self.binary_path,
            "result2msa",
            query_db_path,
            target_db,
            result_db,
            result_a3m_path,
            "--db-load-mode", "0",
            "--msa-format-mode", "6"
        ]
        try:
            run_with_logging(cmd)
            with open(result_a3m_path) as f:
                a3m_content = f.read()
            return msa_tool.MsaToolResult(a3m=a3m_content, target_sequence=target_sequence, e_value=self.e_value)
        except subprocess.CalledProcessError as e:
            logging.error(f"MMseqs2 result2msa failed: {e}")
            return msa_tool.MsaToolResult(a3m="", target_sequence=target_sequence, e_value=self.e_value)
        except IOError as e:
            logging.error(f"Failed to read A3M file {result_a3m_path}: {e}")
            return msa_tool.MsaToolResult(a3m="", target_sequence=target_sequence, e_value=self.e_value)

    def _cpu_search(
        self, 
        query_path: str, 
        result_m8: str, 
        tmp_dir: str,
        target_sequence: str,
    ) -> msa_tool.MsaToolResult:
        """Search using CPU-only MMseqs2."""
        logging.info(f"[DEBUG] _cpu_search called: base_db={self.base_db}")
        query_db = os.path.join(tmp_dir, "query_db")
        result_db = os.path.join(tmp_dir, "result")
        
        cmd = [self.binary_path, "createdb", query_path, query_db]
        run_with_logging(cmd)
        
        cmd = [
            self.binary_path,
            "search",
            query_db,
            self.base_db,
            result_db,
            tmp_dir,
            "--threads", str(self.n_cpu),
            "-e", str(self.e_value),
            "--max-seqs", str(self.max_sequences),
            "-s", str(self.sensitivity),
            "--db-load-mode", "0",
        ]
        try:
            run_with_logging(cmd)
        except subprocess.CalledProcessError as e:
            logging.error(f"CPU search failed for {self.base_db}: {e.stderr if e.stderr else e}")
            return msa_tool.MsaToolResult(a3m="", target_sequence=target_sequence, e_value=self.e_value)

        cmd = [
            self.binary_path,
            "convertalis",
            query_db,
            self.base_db,
            result_db,
            result_m8,
            "--format-mode", "0"
        ]
        try:
            run_with_logging(cmd)
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to convert results to m8 format: {e.stderr if e.stderr else e}")
            return msa_tool.MsaToolResult(a3m="", target_sequence=target_sequence, e_value=self.e_value)

        return self._process_search_results(
            result_db=result_db,
            target_sequence=target_sequence,
            query_db_path=query_db,
            tmp_dir=tmp_dir
        )

    def query(self, target_sequence: str) -> msa_tool.MsaToolResult:
        """Search sequence database using MMseqs2."""
        logging.info("Query sequence: %s", target_sequence[:50] + "..." if len(target_sequence) > 50 else target_sequence)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            query_path = os.path.join(tmp_dir, "query.fasta")
            result_m8 = os.path.join(tmp_dir, "result.m8")
            
            with open(query_path, "w") as f:
                f.write(f">query\n{target_sequence}\n")
            
            try:
                nvidia_smi = "nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader"
                result = subprocess.run(
                    nvidia_smi.split(),
                    capture_output=True,
                    check=True,
                    text=True
                )
                gpu_ids = result.stdout.strip().split("\n")
                if not gpu_ids or gpu_ids == [""]:
                    logging.warning("No GPU devices found")
                    return self._cpu_search(query_path, result_m8, tmp_dir, target_sequence)
                logging.info(f"Found {len(gpu_ids)} GPU devices")
            except (subprocess.SubprocessError, FileNotFoundError):
                logging.warning("Failed to get GPU information")
                return self._cpu_search(query_path, result_m8, tmp_dir, target_sequence)

            gpu_db = self.gpu_db
            logging.info(f"[DEBUG] query: use_gpu={self.use_gpu}, gpu_db={gpu_db}")
            
            if gpu_db:
                logging.info("[DEBUG] Using hybrid GPU search (GPU prefilter + CPU realign)")
                return self._gpu_search(query_path, result_m8, tmp_dir, target_sequence)
            
            logging.info("[DEBUG] GPU database not available, using CPU search")
            return self._cpu_search(query_path, result_m8, tmp_dir, target_sequence)
