import unittest
from unittest.mock import patch, MagicMock
import os
import sys

import app


class TestAppAuthentication(unittest.TestCase):
    def test_is_test_environment(self):
        self.assertTrue(app.is_test_environment())

    def test_check_app_access_in_test_env_returns_true(self):
        self.assertTrue(app.check_app_access())

    @patch("app.is_test_environment", return_value=False)
    @patch("streamlit.session_state", {"is_authenticated": True})
    def test_authenticated_session_returns_true(self, mock_test_env):
        self.assertTrue(app.check_app_access())

    @patch("app.is_test_environment", return_value=False)
    @patch("streamlit.session_state", {"is_authenticated": False})
    @patch("streamlit.query_params", {})
    def test_unauthenticated_session_renders_gate_and_returns_false(self, mock_test_env):
        with patch("streamlit.markdown") as mock_md, \
             patch("streamlit.columns", return_value=[MagicMock(), MagicMock(), MagicMock()]), \
             patch("streamlit.form", return_value=MagicMock()), \
             patch("streamlit.text_input", return_value=""), \
             patch("streamlit.form_submit_button", return_value=False):
            result = app.check_app_access()
            self.assertFalse(result)

    @patch("app.is_test_environment", return_value=False)
    @patch("streamlit.session_state", {"is_authenticated": False})
    @patch("streamlit.query_params", {"token": "500djrqnwk"})
    def test_query_token_auto_login(self, mock_test_env):
        with patch("streamlit.rerun") as mock_rerun:
            result = app.check_app_access()
            self.assertTrue(app.st.session_state.get("is_authenticated"))
            mock_rerun.assert_called_once()


if __name__ == "__main__":
    unittest.main()
