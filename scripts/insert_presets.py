from app.models.database.database import Session

import argparse
from app.models.database.database import Session
from app.models.database.agent_preset import AgentPreset
import datetime

def main(to_commit: bool):

    default_preset_exists = Session.query(AgentPreset).filter(AgentPreset.name == "Default Preset").first() is not None

    if not default_preset_exists:
        default_preset = AgentPreset(
            name="Default Preset",
            version="1.0",
            description="This is the default preset for new agents.",
            max_tokens=1024,
            n=1,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            tags="default,general",
            working_context_length=1024,
            long_term_context_length=4096,
            agent_type="general",
            prompt="Hello, how can I assist you today?",
            interactive_only=False,
            create_date=datetime.datetime.now(),
            update_date=datetime.datetime.now()
        )
    
        # Add the dummy preset to the session
        Session.add(default_preset)

        # Commit the transaction if the --commit flag is provided
        if to_commit:
            Session.commit()
            print("Preset added to the database.")
        else:
            # Rollback the transaction to avoid inserting dummy data
            Session.rollback()
            print("Transaction rolled back. No changes were committed to the database.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert dummy data into the database.")
    parser.add_argument("--commit", action="store_true", help="Commit the changes to the database.")
    
    args = parser.parse_args()
    
    main(args.commit)
