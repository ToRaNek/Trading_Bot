"""
Debug de l'API Claude pour comprendre pourquoi elle retourne 50.0
"""

import asyncio
import os
import aiohttp
from dotenv import load_dotenv
import json
import re

load_dotenv()


async def test_claude_simple():
    """Test simple de l'API Claude"""

    api_key = os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        print("ERREUR: Pas de cle ANTHROPIC_API_KEY dans .env")
        return

    print("=" * 80)
    print("TEST SIMPLE DE L'API CLAUDE")
    print("=" * 80)
    print(f"\nCle API: {api_key[:20]}...{api_key[-10:]}")

    # Test 1: Prompt très simple
    print("\n[Test 1] Prompt simple...")

    prompt = """Score this news sentiment from 0-100:

"Amazon invests $3 billion in new data center"

Reply ONLY with JSON: {"score": <number>}"""

    async with aiohttp.ClientSession() as session:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        payload = {
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": prompt
            }]
        }

        print(f"Modele: {payload['model']}")
        print(f"Max tokens: {payload['max_tokens']}")

        try:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=30
            ) as response:

                status = response.status
                print(f"\nStatus HTTP: {status}")

                if status == 200:
                    result = await response.json()

                    # Afficher la réponse complète
                    print("\n--- REPONSE COMPLETE ---")
                    print(json.dumps(result, indent=2))

                    # Extraire le contenu
                    if 'content' in result:
                        content_blocks = result['content']
                        if content_blocks and len(content_blocks) > 0:
                            text = content_blocks[0].get('text', '')
                            print(f"\n--- TEXTE EXTRAIT ---")
                            print(text)

                            # Parser le JSON
                            json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(0)
                                print(f"\n--- JSON TROUVE ---")
                                print(json_str)

                                try:
                                    data = json.loads(json_str)
                                    score = data.get('score', 'N/A')
                                    print(f"\n--- SCORE EXTRAIT ---")
                                    print(f"Score: {score}")
                                except json.JSONDecodeError as e:
                                    print(f"ERREUR JSON: {e}")
                            else:
                                print("\nAUCUN JSON TROUVE dans la reponse")
                        else:
                            print("ERREUR: content vide")
                    else:
                        print("ERREUR: Pas de 'content' dans la reponse")

                else:
                    error_text = await response.text()
                    print(f"\nERREUR HTTP {status}:")
                    print(error_text)

        except Exception as e:
            print(f"\nEXCEPTION: {e}")
            import traceback
            traceback.print_exc()


async def test_claude_with_real_news():
    """Test avec de vraies news AMZN"""

    api_key = os.getenv('ANTHROPIC_API_KEY')

    print("\n" + "=" * 80)
    print("TEST AVEC VRAIES NEWS AMZN")
    print("=" * 80)

    news_sample = [
        "Amazon invests $3 billion in new Mississippi data center",
        "Amazon Web Services expands cloud infrastructure",
        "Tech stocks show mixed results amid market concerns"
    ]

    news_text = "\n".join([f"{i+1}. {news}" for i, news in enumerate(news_sample)])

    prompt = f"""Analyze these Amazon news and give a sentiment score 0-100.

NEWS:
{news_text}

Reply ONLY with JSON format: {{"score": <number 0-100>}}"""

    async with aiohttp.ClientSession() as session:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        payload = {
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 200,
            "temperature": 0.5,
            "messages": [{
                "role": "user",
                "content": prompt
            }]
        }

        print(f"\n[Prompt envoye]")
        print(prompt[:200] + "...")

        try:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=30
            ) as response:

                if response.status == 200:
                    result = await response.json()

                    content = result.get('content', [{}])[0].get('text', '')
                    print(f"\n[Reponse Claude]")
                    print(content)

                    # Parser
                    json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(0))
                        score = data.get('score', 'N/A')
                        print(f"\n[Score final] {score}/100")
                    else:
                        print("\n[ERREUR] Pas de JSON trouve")

                else:
                    error = await response.text()
                    print(f"\n[ERREUR HTTP {response.status}]")
                    print(error)

        except Exception as e:
            print(f"\n[EXCEPTION] {e}")
            import traceback
            traceback.print_exc()


async def main():
    await test_claude_simple()
    await asyncio.sleep(2)
    await test_claude_with_real_news()


if __name__ == "__main__":
    asyncio.run(main())
