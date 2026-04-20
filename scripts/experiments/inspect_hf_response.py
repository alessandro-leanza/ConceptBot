import json
from gradio_client import Client

client = Client("cstr/conceptnet_normalized")

word = "apple"
relations = [
    "IsA","RelatedTo","PartOf","HasA","UsedFor",
    "CapableOf","Synonym","Antonym","AtLocation","HasProperty","MadeOf"
]

result = client.predict(
    word=word,
    lang="en",
    selected_relations=relations,
    api_name="/get_semantic_profile"
)

print("RAW RESULT TYPE:", type(result))
print("RAW RESULT:")
print(result)

# If JSON-serializable, dump to file for inspection
try:
    with open("/tmp/hf_conceptnet_response.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved to /tmp/hf_conceptnet_response.json")
except Exception as e:
    print("Could not JSON-dump result:", e)
