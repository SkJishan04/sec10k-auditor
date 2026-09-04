# 📊 SEC 10-K RAG & Financial Statement Auditor

**Agentic Hybrid-RAG system for automated financial risk detection in SEC 10-K filings**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-blue.svg)](#testing)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](#docker)

<!-- 📸 IMAGE PLACEHOLDER 1: Hero banner / architecture illustration -->
<!-- ![Project Banner](docs/images/banner.png) -->

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Motivation](#motivation)
4. [Key Features](#key-features)
5. [System Workflow](#system-workflow)
6. [Architecture](#architecture)
7. [Tech Stack](#tech-stack)
8. [Project Structure](#project-structure)
9. [Usage Examples](#usage-examples)
10. [Results](#results)
11. [Evaluation Methodology](#evaluation-methodology)
12. [Setup & Installation](#setup--installation)
13. [Environment Variables](#environment-variables)
14. [Testing](#testing)
15. [Docker](#docker)
16. [CI/CD](#cicd)
17. [Limitations](#limitations)
18. [Future Improvements](#future-improvements)
19. [License](#license)

---

## Overview

This project is a **production-oriented Retrieval-Augmented Generation (RAG) system** that automates one of the most time-consuming tasks in financial due diligence: extracting and cross-checking numerical disclosures buried inside SEC 10-K filings.

It combines **hybrid retrieval** (dense semantic search + sparse BM25 keyword search), a **DPO-fine-tuned Llama-3 8B model** specialized for numeric extraction, and a custom **hallucination guard** that verifies every extracted figure against its retrieved source text before it is ever surfaced to a user.

The result is a system that produces a structured, source-cited **Financial Risk Report** for a given filing — flagging things like off-balance-sheet liabilities, aggressive revenue recognition, and going-concern language — with every numeric claim traceable back to a specific page.

## Problem Statement

Financial analysts, auditors, and M&A due-diligence teams routinely spend **hundreds of hours per deal** manually reading 10-K filings that can run 150–300+ pages. Key risks are often buried in footnotes:

- Off-balance-sheet financing arrangements
- Revenue recognition policies that front-load earnings
- Related-party transactions
- Contingent liabilities and litigation reserves
- Going-concern qualifications

Manual review is slow, inconsistent across analysts, and error-prone at scale — and naive LLM summarization introduces a worse failure mode: **numeric hallucination**, where a model confidently reports a dollar figure that doesn't actually appear in the source document.

## Motivation

> *"Financial due diligence in M&A takes weeks due to manual parsing of SEC 10-K footnoted risks. This system was built to cut financial-audit extraction time by automating retrieval and extraction — while treating numeric hallucination as a hard failure mode to engineer against, not an acceptable tradeoff."*

This project exists to demonstrate that LLM systems can be built **responsibly** in high-stakes, numeric-sensitive domains — by pairing generation with retrieval-grounded verification rather than trusting model output at face value.