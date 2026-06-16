# --- history.py ---
import pandas as pd

class HistoryManager:
    def __init__(self):
        self.undo_stack = []

    def record(self, action, index, original_data, additional_info):
        print(f"Recording action: {action}, Index: {index}")
        self.undo_stack.append({
            "action": action,
            "index": index,
            "original_data": original_data,
            "additional_info": additional_info
        })

    def undo(self, peak_results_df):
        """Restores the last change from the undo stack."""
        if not self.undo_stack:
            print("No actions to undo.")
            return peak_results_df, None

        action, index, old_data, additional_info = self.undo_stack.pop()
        print(f"Undoing action: {action}, Index: {index}")  # Debugging

        if action == "reject":
            # Restore the rejected peak using pd.concat
            peak_results_df = pd.concat([peak_results_df, old_data.to_frame().T], ignore_index=True)
        elif action == "move_base":
            # Restore the base value (example logic; adjust as needed)
            peak_results_df.at[index, 'base_value'] = additional_info
        else:
            print(f"Unknown action {action}")
        
        return peak_results_df, index
