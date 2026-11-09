import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from src.config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
)


# Build database connection information
connection_info = psycopg.conninfo.make_conninfo(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
)


# Create a reusable connection pool
connection_pool = ConnectionPool(
    conninfo=connection_info,
    min_size=1,
    max_size=10,
    configure=register_vector,
)


def get_connection():
    return connection_pool.connection()


if __name__ == "__main__":
    try:
        with get_connection() as connection:
            print("PostgreSQL connection successful!")

        print("Connection pool is working!")

    except Exception as error:
        print(f"Connection failed: {error}")

    finally:
        connection_pool.close()