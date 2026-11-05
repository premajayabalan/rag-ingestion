from src.database.connection import get_connection


def check_pgvector():
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT extname, extversion
                FROM pg_extension
                WHERE extname = 'vector';
            """)

            result = cursor.fetchone()

            if result:
                print(f"pgvector enabled: version {result[1]}")
            else:
                print("pgvector is not enabled in this database.")


if __name__ == "__main__":
    check_pgvector()