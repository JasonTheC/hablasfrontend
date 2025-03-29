import sqlite3
from pathlib import Path
import shutil
from datetime import datetime

def backup_database(db_path):
    """Create a backup of the database before conversion"""
    backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(db_path, backup_path)
    print(f"Created backup at: {backup_path}")
    return backup_path

def convert_database(db_path):
    """Convert the database schema from page to CFI"""
    
    # Create backup first
    backup_path = backup_database(db_path)
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("Starting database conversion...")
        
        # Check if 'page' column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'page' in columns:
            print("Found 'page' column, starting conversion...")
            
            # Create temporary table with new schema
            cursor.execute('''
                CREATE TABLE users_new (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL,
                    current_book TEXT DEFAULT '',
                    cfi TEXT DEFAULT '',
                    utterrance_fname TEXT DEFAULT '',
                    predicted_sentence TEXT DEFAULT '',
                    preferred_language TEXT DEFAULT 'en',
                    login_token TEXT,
                    token_expiry TIMESTAMP
                )
            ''')
            
            # Copy data from old table to new table
            cursor.execute('''
                INSERT INTO users_new (
                    username,
                    password,
                    current_book,
                    cfi,
                    utterrance_fname,
                    predicted_sentence,
                    preferred_language,
                    login_token,
                    token_expiry
                )
                SELECT
                    username,
                    password,
                    current_book,
                    page,  -- Copy page data to cfi column
                    utterrance_fname,
                    predicted_sentence,
                    preferred_language,
                    login_token,
                    token_expiry
                FROM users
            ''')
            
            # Drop old table
            cursor.execute('DROP TABLE users')
            
            # Rename new table to users
            cursor.execute('ALTER TABLE users_new RENAME TO users')
            
            # Commit changes
            conn.commit()
            print("Successfully converted database schema")
            
            # Verify the conversion
            cursor.execute("PRAGMA table_info(users)")
            new_columns = [column[1] for column in cursor.fetchall()]
            print("\nNew database schema:")
            for column in new_columns:
                print(f"- {column}")
            
            # Count records
            cursor.execute("SELECT COUNT(*) FROM users")
            count = cursor.fetchone()[0]
            print(f"\nTotal records in converted database: {count}")
            
        else:
            print("Database already using new schema (no 'page' column found)")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        print(f"Rolling back to backup from: {backup_path}")
        shutil.copy2(backup_path, db_path)
        return False
    finally:
        conn.close()
    
    return True

if __name__ == "__main__":
    DB_PATH = "usersHablas.db"  # Update this to your database path
    
    if Path(DB_PATH).exists():
        print(f"Found database at: {DB_PATH}")
        success = convert_database(DB_PATH)
        if success:
            print("\nDatabase conversion completed successfully!")
            print("You may want to keep the backup file for a while in case of issues.")
        else:
            print("\nDatabase conversion failed! Restored from backup.")
            print("Please check the error messages above.")
    else:
        print(f"Database not found at: {DB_PATH}") 