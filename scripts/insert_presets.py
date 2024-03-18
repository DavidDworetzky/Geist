from app.models.database.database import Session

import argparse
from app.models.database.database import Session
from app.models.database.agent_preset import AgentPreset

def main(to_commit: bool):

    default_preset_exists = Session.query(AgentPreset).filter(AgentPreset.name == "Default Preset").first() is not None

    if not default_preset_exists:
        # Create a dummy AgentPreset object
        default_preset = AgentPreset(
            max_tokens=16,
            n=1,
            temperature=1.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            agent_preset_id=1,
            creation_date="2022-01-01",
            last_modified="2022-01-01",
            version=1,
            status="active",
            configuration={"key": "value"},  # Assuming configuration is a dictionary
            name="Default Preset",
            description="This is a dummy preset for demonstration purposes.",
            restrictions=[]
        )
    
        # Add the dummy preset to the session
        Session.add(default_preset)

        # Commit the transaction if the --commit flag is provided
        if to_commit:
            Session.commit()
            print("Dummy data inserted and committed to the database.")
        else:
            # Rollback the transaction to avoid inserting dummy data
            Session.rollback()
            print("Transaction rolled back. No changes were committed to the database.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert dummy data into the database.")
    parser.add_argument("--commit", action="store_true", help="Commit the changes to the database.")
    
    args = parser.parse_args()
    
    main(args.commit)
