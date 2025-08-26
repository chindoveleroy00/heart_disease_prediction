# Debug script to check authentication issues
import sqlite3
import hashlib
from pathlib import Path

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def debug_authentication():
    """Debug authentication issues"""
    
    # Connect to database
    db_path = Path("database/heart_disease.db")
    if not db_path.exists():
        print("❌ Database file not found!")
        return
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Check if users table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    if not cursor.fetchone():
        print("❌ Users table doesn't exist!")
        return
    
    # Get all users from database
    print("🔍 Checking users in database...")
    cursor.execute("SELECT id, username, email, role, password_hash FROM users")
    users = cursor.fetchall()
    
    if not users:
        print("❌ No users found in database!")
        return
    
    print(f"✅ Found {len(users)} users:")
    for user in users:
        print(f"  ID: {user[0]}, Username: '{user[1]}', Email: '{user[2]}', Role: '{user[3]}'")
        print(f"  Password Hash: {user[4][:20]}...")
    
    print("\n" + "="*50)
    
    # Test authentication with each user
    for user in users:
        username = user[1]
        stored_hash = user[4]
        
        print(f"\n🧪 Testing user: {username}")
        
        # Try common default passwords
        test_passwords = ["admin123", "password", "123456", "admin", username]
        
        for test_pass in test_passwords:
            test_hash = hash_password(test_pass)
            if test_hash == stored_hash:
                print(f"✅ FOUND PASSWORD for {username}: '{test_pass}'")
                print(f"   Hash matches: {test_hash}")
                break
        else:
            print(f"❌ No matching password found for {username}")
            print(f"   Stored hash: {stored_hash}")
            print(f"   Try manual password reset...")
    
    conn.close()

def reset_user_password(role):
    """Reset user password for specific role to known value"""
    
    db_path = Path("database/heart_disease.db")
    if not db_path.exists():
        print("❌ Database file not found!")
        return
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Default passwords for each role
    default_passwords = {
        'admin': 'admin123',
        'clinician': 'clinic123',
        'analyst': 'analyst123'
    }
    
    new_password = default_passwords.get(role, f"{role}123")
    new_hash = hash_password(new_password)
    
    cursor.execute("UPDATE users SET password_hash = ? WHERE role = ?", (new_hash, role))
    affected = cursor.rowcount
    
    if affected > 0:
        conn.commit()
        print(f"✅ Reset password for {affected} {role} user(s)")
        print(f"   New password: '{new_password}'")
        print(f"   New hash: {new_hash[:20]}...")
    else:
        print(f"❌ No {role} users found to reset")
    
    conn.close()

def create_test_user(role):
    """Create a test user with known credentials for specific role"""
    
    db_path = Path("database/heart_disease.db")
    if not db_path.exists():
        print("❌ Database file not found!")
        return
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # User configurations for each role
    user_configs = {
        'admin': {
            'username': 'testadmin',
            'email': 'testadmin@hospital.com',
            'password': 'admin123'
        },
        'clinician': {
            'username': 'testclinician', 
            'email': 'testclinician@hospital.com',
            'password': 'clinic123'
        },
        'analyst': {
            'username': 'testanalyst',
            'email': 'testanalyst@hospital.com', 
            'password': 'analyst123'
        }
    }
    
    if role not in user_configs:
        print(f"❌ Invalid role: {role}")
        return
    
    config = user_configs[role]
    username = config['username']
    
    # Check if test user already exists
    cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        print(f"⚠️ Test {role} '{username}' already exists")
        conn.close()
        return
    
    # Create test user
    email = config['email']
    password = config['password']
    password_hash = hash_password(password)
    
    from datetime import datetime
    
    cursor.execute(
        "INSERT INTO users (username, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
        (username, email, password_hash, role, datetime.now())
    )
    
    conn.commit()
    print(f"✅ Created test {role} user:")
    print(f"   Username: '{username}'")
    print(f"   Password: '{password}'")
    print(f"   Email: '{email}'")
    print(f"   Role: '{role}'")
    
    conn.close()

def create_all_test_users():
    """Create test users for all roles"""
    
    print("🔧 Creating test users for all roles...")
    roles = ['admin', 'clinician', 'analyst']
    
    for role in roles:
        print(f"\n📝 Creating {role} user...")
        create_test_user(role)

def reset_all_passwords():
    """Reset passwords for all existing users"""
    
    print("🔐 Resetting passwords for all user roles...")
    roles = ['admin', 'clinician', 'analyst']
    
    for role in roles:
        print(f"\n🔄 Resetting {role} passwords...")
        reset_user_password(role)

if __name__ == "__main__":
    print("🔐 Authentication Debug Tool - All User Roles")
    print("="*50)
    
    # Run debug
    debug_authentication()
    
    print("\n" + "="*50)
    print("🔧 Choose an action:")
    print("1. Reset ALL user passwords (admin123, clinic123, analyst123)")
    print("2. Create test users for ALL roles")
    print("3. Reset ADMIN password only")
    print("4. Reset CLINICIAN password only") 
    print("5. Reset ANALYST password only")
    print("6. Create test ADMIN user")
    print("7. Create test CLINICIAN user")
    print("8. Create test ANALYST user")
    print("9. Show current user summary")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-9): ").strip()
    
    if choice == "1":
        reset_all_passwords()
    elif choice == "2":
        create_all_test_users()
    elif choice == "3":
        reset_user_password('admin')
    elif choice == "4":
        reset_user_password('clinician')
    elif choice == "5":
        reset_user_password('analyst')
    elif choice == "6":
        create_test_user('admin')
    elif choice == "7":
        create_test_user('clinician')
    elif choice == "8":
        create_test_user('analyst')
    elif choice == "9":
        debug_authentication()
    elif choice == "0":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")
    
    if choice != "0" and choice != "9":
        print("\n" + "="*50)
        print("🔄 Running final check...")
        debug_authentication()
        
        print("\n📋 User Summary:")
        print("Admin users    - Password: admin123")
        print("Clinician users - Password: clinic123") 
        print("Analyst users   - Password: analyst123")