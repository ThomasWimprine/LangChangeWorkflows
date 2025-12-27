"""
Tests for Auto-Bootstrap Embedding Cache (PRP-012).

Tests cover all 7 tasks:
- 012-001: Cache Detection
- 012-002: Smart File Discovery
- 012-003: Embedding Creation
- 012-004: Graceful Fallback Logic
- 012-005: Security Controls
- 012-006: Workflow Integration
- 012-007: Gitignore Update
"""

import json
import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from workflows.retrieval.bootstrap import (
    MAX_CONTENT_CHARS,
    MAX_FILES_TO_EMBED,
    MIN_EMBEDDINGS_THRESHOLD,
    _check_bootstrap_prerequisites,
    _create_embeddings,
    _discover_code_files,
    _ensure_embedding_cache,
    _has_sufficient_cache,
    _is_excluded_path,
    _update_gitignore,
)


# ============================================================================
# TASK 012-001: CACHE DETECTION TESTS
# ============================================================================


class TestCacheDetection:
    """Tests for _has_sufficient_cache function."""

    def test_returns_false_when_cache_missing(self):
        """Returns False when .emb_cache/ directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            assert _has_sufficient_cache(project_path) is False

    def test_returns_false_when_cache_empty(self):
        """Returns False when .emb_cache/ exists but is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            cache_dir = project_path / ".emb_cache"
            cache_dir.mkdir()
            assert _has_sufficient_cache(project_path) is False

    def test_returns_false_when_cache_sparse(self):
        """Returns False when .emb_cache/ has <= MIN_EMBEDDINGS_THRESHOLD files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            cache_dir = project_path / ".emb_cache"
            cache_dir.mkdir()

            # Create exactly threshold number of files
            for i in range(MIN_EMBEDDINGS_THRESHOLD):
                (cache_dir / f"chunk_{i}.json").write_text("[]")

            assert _has_sufficient_cache(project_path) is False

    def test_returns_true_when_cache_sufficient(self):
        """Returns True when .emb_cache/ has > MIN_EMBEDDINGS_THRESHOLD files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            cache_dir = project_path / ".emb_cache"
            cache_dir.mkdir()

            # Create more than threshold number of files
            for i in range(MIN_EMBEDDINGS_THRESHOLD + 1):
                (cache_dir / f"chunk_{i}.json").write_text("[]")

            assert _has_sufficient_cache(project_path) is True

    def test_ignores_meta_json_files(self):
        """Meta files (.meta.json) are not counted as embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            cache_dir = project_path / ".emb_cache"
            cache_dir.mkdir()

            # Create 5 embeddings + 5 meta files = still only 5 actual embeddings
            for i in range(5):
                (cache_dir / f"chunk_{i}.json").write_text("[]")
                (cache_dir / f"chunk_{i}.meta.json").write_text("{}")

            assert _has_sufficient_cache(project_path) is False

    def test_handles_permission_error_gracefully(self):
        """Returns False and logs warning on permission error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            cache_dir = project_path / ".emb_cache"
            cache_dir.mkdir()

            # Remove read permission
            os.chmod(cache_dir, 0o000)

            try:
                result = _has_sufficient_cache(project_path)
                # Should return False without raising exception
                assert result is False
            finally:
                # Restore permissions for cleanup (owner-only is sufficient)
                os.chmod(cache_dir, stat.S_IRWXU)  # 0o700

    def test_returns_false_when_cache_is_file_not_dir(self):
        """Returns False when .emb_cache is a file instead of directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            cache_file = project_path / ".emb_cache"
            cache_file.write_text("not a directory")

            assert _has_sufficient_cache(project_path) is False


# ============================================================================
# TASK 012-002: FILE DISCOVERY TESTS
# ============================================================================


class TestFileDiscovery:
    """Tests for _discover_code_files function."""

    def test_discovers_python_files(self):
        """Discovers .py files in project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "main.py").write_text("print('hello')")
            (project_path / "utils.py").write_text("def helper(): pass")

            files = _discover_code_files(project_path)
            assert len(files) == 2
            assert all(f.suffix == ".py" for f in files)

    def test_prioritizes_src_over_root(self):
        """Files in src/ come before files in root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "root.py").write_text("root")
            src_dir = project_path / "src"
            src_dir.mkdir()
            (src_dir / "core.py").write_text("core")

            files = _discover_code_files(project_path)
            # src/core.py should come first
            assert files[0].name == "core.py"

    def test_excludes_node_modules(self):
        """Files in node_modules/ are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "app.js").write_text("app")
            nm_dir = project_path / "node_modules" / "pkg"
            nm_dir.mkdir(parents=True)
            (nm_dir / "index.js").write_text("pkg")

            files = _discover_code_files(project_path)
            assert len(files) == 1
            assert files[0].name == "app.js"

    def test_excludes_venv(self):
        """Files in venv/ and .venv/ are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "main.py").write_text("main")

            venv_dir = project_path / "venv" / "lib"
            venv_dir.mkdir(parents=True)
            (venv_dir / "site.py").write_text("site")

            dot_venv = project_path / ".venv" / "lib"
            dot_venv.mkdir(parents=True)
            (dot_venv / "site.py").write_text("site")

            files = _discover_code_files(project_path)
            assert len(files) == 1
            assert files[0].name == "main.py"

    def test_excludes_pycache(self):
        """Files in __pycache__/ are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "main.py").write_text("main")

            cache_dir = project_path / "__pycache__"
            cache_dir.mkdir()
            (cache_dir / "main.cpython-311.pyc").write_text("compiled")

            files = _discover_code_files(project_path)
            assert len(files) == 1

    def test_excludes_secret_files(self):
        """Files matching secret patterns are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "main.py").write_text("main")
            (project_path / ".env").write_text("SECRET=xyz")
            (project_path / "credentials.json").write_text("{}")
            (project_path / "api_secret.py").write_text("key=...")
            (project_path / "auth_key.py").write_text("key=...")

            files = _discover_code_files(project_path)
            # Only main.py should be discovered
            assert len(files) == 1
            assert files[0].name == "main.py"

    def test_excludes_minified_files(self):
        """Minified JS files are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "app.js").write_text("app")
            (project_path / "app.min.js").write_text("minified")
            (project_path / "vendor.bundle.js").write_text("bundled")

            files = _discover_code_files(project_path)
            assert len(files) == 1
            assert files[0].name == "app.js"

    def test_limits_to_max_files(self):
        """Limits results to MAX_FILES_TO_EMBED."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create more files than the limit
            for i in range(MAX_FILES_TO_EMBED + 50):
                (project_path / f"file_{i:04d}.py").write_text(f"# file {i}")

            files = _discover_code_files(project_path)
            assert len(files) == MAX_FILES_TO_EMBED

    def test_discovers_multiple_extensions(self):
        """Discovers files with various code extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "main.py").write_text("python")
            (project_path / "app.js").write_text("javascript")
            (project_path / "app.ts").write_text("typescript")
            (project_path / "main.go").write_text("golang")
            (project_path / "Main.java").write_text("java")
            (project_path / "main.rs").write_text("rust")

            files = _discover_code_files(project_path)
            assert len(files) == 6

    def test_skips_symlinks(self):
        """Symlinks are skipped to prevent infinite loops."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            real_file = project_path / "real.py"
            real_file.write_text("real")

            link_file = project_path / "link.py"
            link_file.symlink_to(real_file)

            files = _discover_code_files(project_path)
            # Only the real file, not the symlink
            assert len(files) == 1
            assert files[0].name == "real.py"

    def test_handles_empty_project(self):
        """Returns empty list for project with no code files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "README.md").write_text("readme")
            (project_path / "data.json").write_text("{}")

            files = _discover_code_files(project_path)
            assert files == []


class TestIsExcludedPath:
    """Tests for _is_excluded_path helper."""

    def test_excludes_node_modules(self):
        """Paths containing node_modules are excluded."""
        project = Path("/project")
        path = Path("/project/node_modules/pkg/index.js")
        assert _is_excluded_path(path, project) is True

    def test_excludes_env_file(self):
        """Files named .env are excluded."""
        project = Path("/project")
        path = Path("/project/.env")
        assert _is_excluded_path(path, project) is True

    def test_excludes_credentials(self):
        """Files with 'credentials' in name are excluded."""
        project = Path("/project")
        path = Path("/project/api_credentials.json")
        assert _is_excluded_path(path, project) is True

    def test_allows_normal_files(self):
        """Normal code files are not excluded."""
        project = Path("/project")
        path = Path("/project/src/main.py")
        assert _is_excluded_path(path, project) is False


# ============================================================================
# TASK 012-003: EMBEDDING CREATION TESTS
# ============================================================================


class TestEmbeddingCreation:
    """Tests for _create_embeddings function."""

    def test_creates_embeddings_with_mock_api(self):
        """Creates embeddings using mocked OpenAI API."""
        import workflows.retrieval.bootstrap as bootstrap_module

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            cache_dir = project_path / ".emb_cache"

            # Create test file
            test_file = project_path / "test.py"
            test_file.write_text("def hello(): pass")

            # Mock OpenAI response
            mock_embedding = [0.1] * 3072
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=mock_embedding)]

            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai_class = MagicMock(return_value=mock_client)

            # Patch at module level
            original_openai = bootstrap_module.OpenAI
            original_available = bootstrap_module.OPENAI_AVAILABLE
            try:
                bootstrap_module.OpenAI = mock_openai_class
                bootstrap_module.OPENAI_AVAILABLE = True

                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    count = _create_embeddings([test_file], cache_dir, project_path)
            finally:
                bootstrap_module.OpenAI = original_openai
                bootstrap_module.OPENAI_AVAILABLE = original_available

            assert count == 1
            # Verify embedding file was created
            embedding_files = list(cache_dir.glob("*.json"))
            # Should have one embedding + one metadata file
            assert len([f for f in embedding_files if not f.name.endswith(".meta.json")]) == 1

    def test_returns_zero_without_api_key(self):
        """Returns 0 when OPENAI_API_KEY is not set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            cache_dir = project_path / ".emb_cache"
            test_file = project_path / "test.py"
            test_file.write_text("code")

            # Ensure API key is not set
            with patch.dict(os.environ, {}, clear=True):
                if "OPENAI_API_KEY" in os.environ:
                    del os.environ["OPENAI_API_KEY"]

                count = _create_embeddings([test_file], cache_dir, project_path)

            assert count == 0

    def test_truncates_long_content(self):
        """Truncates file content to MAX_CONTENT_CHARS."""
        import workflows.retrieval.bootstrap as bootstrap_module

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            cache_dir = project_path / ".emb_cache"

            # Create file with content longer than limit
            test_file = project_path / "long.py"
            long_content = "x" * (MAX_CONTENT_CHARS + 1000)
            test_file.write_text(long_content)

            mock_embedding = [0.1] * 3072
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=mock_embedding)]

            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai_class = MagicMock(return_value=mock_client)

            original_openai = bootstrap_module.OpenAI
            original_available = bootstrap_module.OPENAI_AVAILABLE
            try:
                bootstrap_module.OpenAI = mock_openai_class
                bootstrap_module.OPENAI_AVAILABLE = True

                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    _create_embeddings([test_file], cache_dir, project_path)

                    # Verify the input was truncated
                    call_args = mock_client.embeddings.create.call_args
                    input_text = call_args.kwargs.get("input") or call_args[1].get("input")
                    assert len(input_text) <= MAX_CONTENT_CHARS
            finally:
                bootstrap_module.OpenAI = original_openai
                bootstrap_module.OPENAI_AVAILABLE = original_available

    def test_continues_on_individual_failure(self):
        """Continues with remaining files if one fails."""
        import workflows.retrieval.bootstrap as bootstrap_module

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            cache_dir = project_path / ".emb_cache"

            # Create two test files
            file1 = project_path / "file1.py"
            file1.write_text("code1")
            file2 = project_path / "file2.py"
            file2.write_text("code2")

            mock_embedding = [0.1] * 3072
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=mock_embedding)]

            call_count = [0]

            def side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise Exception("API error")
                return mock_response

            mock_client = MagicMock()
            mock_client.embeddings.create.side_effect = side_effect
            mock_openai_class = MagicMock(return_value=mock_client)

            original_openai = bootstrap_module.OpenAI
            original_available = bootstrap_module.OPENAI_AVAILABLE
            try:
                bootstrap_module.OpenAI = mock_openai_class
                bootstrap_module.OPENAI_AVAILABLE = True

                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    count = _create_embeddings([file1, file2], cache_dir, project_path)
            finally:
                bootstrap_module.OpenAI = original_openai
                bootstrap_module.OPENAI_AVAILABLE = original_available

            # Should have embedded 1 file (second one succeeded)
            assert count == 1

    def test_skips_empty_files(self):
        """Skips files with no content."""
        import workflows.retrieval.bootstrap as bootstrap_module

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            cache_dir = project_path / ".emb_cache"

            empty_file = project_path / "empty.py"
            empty_file.write_text("")

            mock_client = MagicMock()
            mock_openai_class = MagicMock(return_value=mock_client)

            original_openai = bootstrap_module.OpenAI
            original_available = bootstrap_module.OPENAI_AVAILABLE
            try:
                bootstrap_module.OpenAI = mock_openai_class
                bootstrap_module.OPENAI_AVAILABLE = True

                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    count = _create_embeddings([empty_file], cache_dir, project_path)
            finally:
                bootstrap_module.OpenAI = original_openai
                bootstrap_module.OPENAI_AVAILABLE = original_available

            assert count == 0
            # API should not have been called
            mock_client.embeddings.create.assert_not_called()

    def test_creates_cache_with_700_permissions(self):
        """Creates cache directory with 700 (owner-only) permissions."""
        import workflows.retrieval.bootstrap as bootstrap_module

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            cache_dir = project_path / ".emb_cache"

            test_file = project_path / "test.py"
            test_file.write_text("code")

            mock_embedding = [0.1] * 3072
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=mock_embedding)]

            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai_class = MagicMock(return_value=mock_client)

            original_openai = bootstrap_module.OpenAI
            original_available = bootstrap_module.OPENAI_AVAILABLE
            try:
                bootstrap_module.OpenAI = mock_openai_class
                bootstrap_module.OPENAI_AVAILABLE = True

                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    _create_embeddings([test_file], cache_dir, project_path)
            finally:
                bootstrap_module.OpenAI = original_openai
                bootstrap_module.OPENAI_AVAILABLE = original_available

            # Check directory permissions
            mode = cache_dir.stat().st_mode
            assert mode & 0o777 == 0o700


# ============================================================================
# TASK 012-004: FALLBACK LOGIC TESTS
# ============================================================================


class TestFallbackLogic:
    """Tests for _check_bootstrap_prerequisites function."""

    def test_returns_false_without_api_key(self):
        """Returns (False, reason) when OPENAI_API_KEY not set."""
        with patch.dict(os.environ, {}, clear=True):
            can_bootstrap, reason = _check_bootstrap_prerequisites()
            assert can_bootstrap is False
            assert "OPENAI_API_KEY" in reason

    def test_returns_false_without_openai_package(self):
        """Returns (False, reason) when openai package not installed."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch.dict("sys.modules", {"openai": None}):
                # Import check will fail
                import builtins
                original_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if name == "openai":
                        raise ImportError("No module named 'openai'")
                    return original_import(name, *args, **kwargs)

                with patch.object(builtins, "__import__", mock_import):
                    can_bootstrap, reason = _check_bootstrap_prerequisites()
                    assert can_bootstrap is False
                    assert "openai" in reason.lower()

    def test_returns_true_when_prerequisites_met(self):
        """Returns (True, '') when all prerequisites are met."""
        import workflows.retrieval.bootstrap as bootstrap_module

        original_available = bootstrap_module.OPENAI_AVAILABLE
        try:
            bootstrap_module.OPENAI_AVAILABLE = True
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                can_bootstrap, reason = _check_bootstrap_prerequisites()
                assert can_bootstrap is True
                assert reason == ""
        finally:
            bootstrap_module.OPENAI_AVAILABLE = original_available


# ============================================================================
# TASK 012-005: SECURITY CONTROLS TESTS
# ============================================================================


class TestAdditionalEdgeCases:
    """Additional edge case tests for better coverage."""

    def test_create_embeddings_when_openai_not_available(self):
        """Returns 0 when OpenAI package not available."""
        import workflows.retrieval.bootstrap as bootstrap_module

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            cache_dir = project_path / ".emb_cache"
            test_file = project_path / "test.py"
            test_file.write_text("code")

            original_available = bootstrap_module.OPENAI_AVAILABLE
            try:
                bootstrap_module.OPENAI_AVAILABLE = False

                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    count = _create_embeddings([test_file], cache_dir, project_path)

                assert count == 0
            finally:
                bootstrap_module.OPENAI_AVAILABLE = original_available

    def test_ensure_cache_when_embedding_fails(self):
        """Returns False when all embeddings fail."""
        import workflows.retrieval.bootstrap as bootstrap_module

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "main.py").write_text("code")

            mock_client = MagicMock()
            mock_client.embeddings.create.side_effect = Exception("API error")
            mock_openai_class = MagicMock(return_value=mock_client)

            original_openai = bootstrap_module.OpenAI
            original_available = bootstrap_module.OPENAI_AVAILABLE
            try:
                bootstrap_module.OpenAI = mock_openai_class
                bootstrap_module.OPENAI_AVAILABLE = True

                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    result = _ensure_embedding_cache(project_path)

                assert result is False
            finally:
                bootstrap_module.OpenAI = original_openai
                bootstrap_module.OPENAI_AVAILABLE = original_available

    def test_is_excluded_path_prefix_match(self):
        """Test prefix match pattern like .env.*"""
        project = Path("/project")
        # .env.local should be excluded
        path = Path("/project/.env.local")
        assert _is_excluded_path(path, project) is True

    def test_cache_dir_permission_failure(self):
        """Returns 0 when cache directory can't be created."""
        import workflows.retrieval.bootstrap as bootstrap_module

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            # Create a file where the cache directory should be
            cache_blocker = project_path / ".emb_cache"
            cache_blocker.write_text("I'm a file, not a directory")

            test_file = project_path / "test.py"
            test_file.write_text("code")

            mock_openai_class = MagicMock()

            original_openai = bootstrap_module.OpenAI
            original_available = bootstrap_module.OPENAI_AVAILABLE
            try:
                bootstrap_module.OpenAI = mock_openai_class
                bootstrap_module.OPENAI_AVAILABLE = True

                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    count = _create_embeddings([test_file], cache_blocker, project_path)

                assert count == 0
            finally:
                bootstrap_module.OpenAI = original_openai
                bootstrap_module.OPENAI_AVAILABLE = original_available


class TestSecurityControls:
    """Tests for security requirements (SR-1, SR-2, SR-3)."""

    def test_api_key_never_logged(self):
        """API key should never appear in log output."""
        import io
        import logging
        import workflows.retrieval.bootstrap as bootstrap_module

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            cache_dir = project_path / ".emb_cache"
            test_file = project_path / "test.py"
            test_file.write_text("code")

            api_key = "sk-test-super-secret-key-12345"

            mock_openai_class = MagicMock(side_effect=Exception("Connection failed"))

            original_openai = bootstrap_module.OpenAI
            original_available = bootstrap_module.OPENAI_AVAILABLE
            try:
                bootstrap_module.OpenAI = mock_openai_class
                bootstrap_module.OPENAI_AVAILABLE = True

                log_capture = io.StringIO()
                handler = logging.StreamHandler(log_capture)
                logger = logging.getLogger("workflows.retrieval.bootstrap")
                logger.addHandler(handler)

                try:
                    with patch.dict(os.environ, {"OPENAI_API_KEY": api_key}):
                        _create_embeddings([test_file], cache_dir, project_path)
                finally:
                    logger.removeHandler(handler)

                log_output = log_capture.getvalue()
                # API key should NOT appear in logs
                assert api_key not in log_output
            finally:
                bootstrap_module.OpenAI = original_openai
                bootstrap_module.OPENAI_AVAILABLE = original_available

    def test_secret_files_excluded(self):
        """Files with secret patterns are excluded from embedding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Create secret files
            (project_path / ".env").write_text("SECRET=xyz")
            (project_path / "credentials.json").write_text("{}")
            (project_path / "api_secret.py").write_text("key=...")

            files = _discover_code_files(project_path)
            # No secret files should be discovered
            assert len(files) == 0

    def test_cache_directory_permissions(self):
        """Cache directory is created with 700 permissions."""
        import workflows.retrieval.bootstrap as bootstrap_module

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            cache_dir = project_path / ".emb_cache"
            test_file = project_path / "test.py"
            test_file.write_text("code")

            mock_embedding = [0.1] * 3072
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=mock_embedding)]

            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai_class = MagicMock(return_value=mock_client)

            original_openai = bootstrap_module.OpenAI
            original_available = bootstrap_module.OPENAI_AVAILABLE
            try:
                bootstrap_module.OpenAI = mock_openai_class
                bootstrap_module.OPENAI_AVAILABLE = True

                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    _create_embeddings([test_file], cache_dir, project_path)
            finally:
                bootstrap_module.OpenAI = original_openai
                bootstrap_module.OPENAI_AVAILABLE = original_available

            # Verify 700 permissions (owner read/write/execute only)
            mode = cache_dir.stat().st_mode
            assert stat.S_IMODE(mode) == 0o700


# ============================================================================
# TASK 012-007: GITIGNORE UPDATE TESTS
# ============================================================================


class TestGitignoreUpdate:
    """Tests for _update_gitignore function."""

    def test_adds_entry_to_existing_gitignore(self):
        """Appends .emb_cache/ to existing .gitignore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            gitignore = project_path / ".gitignore"
            gitignore.write_text("*.pyc\nnode_modules/\n")

            result = _update_gitignore(project_path)

            assert result is True
            content = gitignore.read_text()
            assert ".emb_cache/" in content
            # Original content preserved
            assert "*.pyc" in content
            assert "node_modules/" in content

    def test_creates_gitignore_if_missing(self):
        """Creates .gitignore with .emb_cache/ if file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            gitignore = project_path / ".gitignore"

            result = _update_gitignore(project_path)

            assert result is True
            assert gitignore.exists()
            content = gitignore.read_text()
            assert ".emb_cache/" in content

    def test_idempotent_when_already_present(self):
        """Does not add duplicate entry if already present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            gitignore = project_path / ".gitignore"
            gitignore.write_text(".emb_cache/\n")

            result = _update_gitignore(project_path)

            assert result is True
            content = gitignore.read_text()
            # Should only appear once
            assert content.count(".emb_cache") == 1

    def test_handles_entry_without_trailing_slash(self):
        """Recognizes entry without trailing slash as present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            gitignore = project_path / ".gitignore"
            gitignore.write_text(".emb_cache\n")

            result = _update_gitignore(project_path)

            assert result is True
            content = gitignore.read_text()
            # Should not add duplicate
            assert content.count(".emb_cache") == 1

    def test_handles_permission_error(self):
        """Returns False on permission error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            gitignore = project_path / ".gitignore"
            gitignore.write_text("# comment\n")

            # Make file read-only
            os.chmod(gitignore, 0o444)

            try:
                result = _update_gitignore(project_path)
                assert result is False
            finally:
                os.chmod(gitignore, 0o644)

    def test_preserves_content_without_trailing_newline(self):
        """Adds newline before entry if file doesn't end with one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            gitignore = project_path / ".gitignore"
            gitignore.write_text("*.pyc")  # No trailing newline

            _update_gitignore(project_path)

            content = gitignore.read_text()
            lines = content.splitlines()
            assert "*.pyc" in lines
            assert ".emb_cache/" in lines


# ============================================================================
# TASK 012-006: WORKFLOW INTEGRATION TESTS
# ============================================================================


class TestWorkflowIntegration:
    """Tests for _ensure_embedding_cache function."""

    def test_returns_true_when_cache_exists(self):
        """Returns True immediately when cache already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            cache_dir = project_path / ".emb_cache"
            cache_dir.mkdir()

            # Create sufficient embeddings
            for i in range(MIN_EMBEDDINGS_THRESHOLD + 1):
                (cache_dir / f"chunk_{i}.json").write_text("[]")

            result = _ensure_embedding_cache(project_path)
            assert result is True

    def test_returns_false_without_api_key(self):
        """Returns False when API key not set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "main.py").write_text("code")

            with patch.dict(os.environ, {}, clear=True):
                result = _ensure_embedding_cache(project_path)

            assert result is False

    def test_creates_cache_and_updates_gitignore(self):
        """Creates cache and updates .gitignore when starting fresh."""
        import workflows.retrieval.bootstrap as bootstrap_module

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "main.py").write_text("code")

            mock_embedding = [0.1] * 3072
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=mock_embedding)]

            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai_class = MagicMock(return_value=mock_client)

            original_openai = bootstrap_module.OpenAI
            original_available = bootstrap_module.OPENAI_AVAILABLE
            try:
                bootstrap_module.OpenAI = mock_openai_class
                bootstrap_module.OPENAI_AVAILABLE = True

                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    result = _ensure_embedding_cache(project_path)
            finally:
                bootstrap_module.OpenAI = original_openai
                bootstrap_module.OPENAI_AVAILABLE = original_available

            assert result is True
            # Cache should exist
            assert (project_path / ".emb_cache").exists()
            # Gitignore should be updated
            assert (project_path / ".gitignore").exists()
            assert ".emb_cache/" in (project_path / ".gitignore").read_text()

    def test_returns_false_when_no_code_files(self):
        """Returns False when project has no code files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "README.md").write_text("readme")

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = _ensure_embedding_cache(project_path)

            assert result is False

    def test_logs_progress_messages(self):
        """Logs progress messages during bootstrap."""
        import io
        import logging
        import workflows.retrieval.bootstrap as bootstrap_module

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "main.py").write_text("code")

            mock_embedding = [0.1] * 3072
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=mock_embedding)]

            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai_class = MagicMock(return_value=mock_client)

            log_capture = io.StringIO()
            handler = logging.StreamHandler(log_capture)
            handler.setLevel(logging.INFO)
            logger = logging.getLogger("workflows.retrieval.bootstrap")
            original_level = logger.level
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)

            original_openai = bootstrap_module.OpenAI
            original_available = bootstrap_module.OPENAI_AVAILABLE
            try:
                bootstrap_module.OpenAI = mock_openai_class
                bootstrap_module.OPENAI_AVAILABLE = True

                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    _ensure_embedding_cache(project_path)
            finally:
                logger.removeHandler(handler)
                logger.setLevel(original_level)
                bootstrap_module.OpenAI = original_openai
                bootstrap_module.OPENAI_AVAILABLE = original_available

            log_output = log_capture.getvalue()
            # Should contain bootstrap messages
            assert "Bootstrapping" in log_output or "embeddings" in log_output.lower()
