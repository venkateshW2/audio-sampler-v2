

set -ex



pip check
pytest -vv certifi/certifi/tests
exit 0
