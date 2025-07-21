from neo4j import GraphDatabase

uri = "neo4j://127.0.0.1:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "test1234"))

def test_connection():
    with driver.session() as session:
        result = session.run("RETURN 'Connected to Neo4j!' AS message")
        for record in result:
            print(record["message"])

if __name__ == "__main__":
    test_connection()
