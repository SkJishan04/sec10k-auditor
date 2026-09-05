def test_create_and_get_filing(client):
    payload = {
        "company_name": "Example Corp",
        "ticker": "EXMP",
        "cik": "0000320193",
        "fiscal_year": 2023,
        "filing_type": "10-K",
    }

    create_response = client.post("/api/filings", json=payload)
    assert create_response.status_code == 201
    body = create_response.json()
    assert body["status"] == "pending"
    assert body["chunk_count"] == 0

    get_response = client.get(f"/api/filings/{body['id']}")
    assert get_response.status_code == 200
    assert get_response.json()["ticker"] == "EXMP"


def test_get_missing_filing_returns_404(client):
    response = client.get("/api/filings/999999")
    assert response.status_code == 404


def test_list_filings(client):
    client.post(
        "/api/filings",
        json={
            "company_name": "Another Corp",
            "ticker": "ANTC",
            "cik": "1234567",
            "fiscal_year": 2022,
            "filing_type": "10-K",
        },
    )
    response = client.get("/api/filings")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) >= 1