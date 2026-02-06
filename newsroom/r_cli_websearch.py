#!/opt/homebrew/Caskroom/miniconda/base/bin/python3.13
import json
import re
import sys

try:
  from r_cli.skills.websearch_skill import WebSearchSkill
except Exception as exc:
  print(json.dumps({"error": str(exc)}))
  sys.exit(1)


def parse_results(text):
  results = []
  if not text or "Error" in text or "Search error" in text:
    return results
  pattern = re.compile(r"\*\*(.*?)\*\*\n(.*?)\nURL:\s*(\S+)", re.S)
  for title, snippet, url in pattern.findall(text):
    results.append({
      "title": title.strip(),
      "snippet": snippet.strip(),
      "url": url.strip(),
      "source": "r_cli_local"
    })
  return results


def main():
  if len(sys.argv) < 2:
    print(json.dumps({"error": "Missing query"}))
    sys.exit(1)
  query = sys.argv[1]
  num_results = int(sys.argv[2]) if len(sys.argv) > 2 else 5
  skill = WebSearchSkill()
  text = skill.web_search(query=query, num_results=num_results)
  results = parse_results(text)
  print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
  main()
