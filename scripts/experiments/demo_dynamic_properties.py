import argparse

from scripts.modules.dynamic_properties import induce_task_properties


BASE_PROPERTIES = ["dangerous", "fragile", "deformable", "hold liquid", "safe", "stable", "poisonous"]
INSTRUCTION = "Heat my food in the microwave."
OBJECTS = ["aluminum tray", "glass container", "soup bowl", "microwave oven"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo lightweight Dynamic Property Induction for generic OPE.")
    parser.add_argument(
        "--run-ope",
        action="store_true",
        help="Also pass induced properties into generic OPE. This may call ConceptNet and the LLM.",
    )
    args = parser.parse_args()

    result = induce_task_properties(
        user_instruction=INSTRUCTION,
        detected_objects=OBJECTS,
        base_properties=BASE_PROPERTIES,
        task_category="microwave_demo",
        max_properties=6,
    )

    dynamic_names = [item["name"] for item in result["dynamic_properties"]]

    print("Instruction:")
    print(INSTRUCTION)
    print("\nObjects:")
    print(OBJECTS)
    print("\nBase properties:")
    print(BASE_PROPERTIES)
    print("\nInduced dynamic properties:")
    print(dynamic_names)
    print("\nMerged properties:")
    print(result["merged_properties"])
    print("\nInduction source:")
    print(result["source"])

    if result["rejected_properties"]:
        print("\nRejected properties:")
        print(result["rejected_properties"])

    if args.run_ope:
        from scripts.modules.ope import OPE

        print("\nRunning generic OPE with induced dynamic properties...")
        objects_info = OPE(
            found_objects=OBJECTS,
            rel_objects=OBJECTS,
            dynamic_properties=dynamic_names,
            dynamic_property_metadata=result,
        )
        print("\nOPE objects_info:")
        print(objects_info)


if __name__ == "__main__":
    main()
