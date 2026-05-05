from __future__ import annotations

import argparse
import csv
from pathlib import Path

from config import neo4j_config


def get_driver(config=None):
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install with `python3 -m pip install -r requirements.txt`.") from exc

    config = config or neo4j_config()
    return GraphDatabase.driver(config.uri, auth=(config.user, config.password))


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.reader(file, delimiter="\t"):
            if row:
                yield row


def import_graph(dataset_dir: Path = Path("dataset/network"), clear: bool = True) -> None:
    driver = get_driver()
    with driver.session() as session:
        if clear:
            session.run("MATCH (n) DETACH DELETE n").consume()

        session.run("CREATE CONSTRAINT restaurant_id IF NOT EXISTS FOR (r:Restaurant) REQUIRE r.id IS UNIQUE").consume()
        session.run("CREATE CONSTRAINT address_id IF NOT EXISTS FOR (a:Address) REQUIRE a.id IS UNIQUE").consume()
        session.run("CREATE CONSTRAINT city_id IF NOT EXISTS FOR (c:City) REQUIRE c.id IS UNIQUE").consume()

        restaurants = [
            {"id": r[0], "label": r[1], "name": r[2], "phone": r[3], "cuisine": r[4]}
            for r in rows(dataset_dir / "restaurant.txt")
            if len(r) >= 5
        ]
        addresses = [{"id": r[0], "label": r[1], "value": r[2]} for r in rows(dataset_dir / "address.txt") if len(r) >= 3]
        cities = [{"id": r[0], "label": r[1], "value": r[2]} for r in rows(dataset_dir / "city.txt") if len(r) >= 3]
        restaurant_address = [
            {"restaurant_id": r[0], "address_id": r[1]}
            for r in rows(dataset_dir / "restaurant-address.txt")
            if len(r) >= 2
        ]
        address_city = [
            {"address_id": r[0], "city_id": r[1]}
            for r in rows(dataset_dir / "address-city.txt")
            if len(r) >= 2
        ]

        session.run(
            """
            UNWIND $rows AS row
            MERGE (r:Restaurant {id: row.id})
            SET r.label = row.label, r.name = row.name, r.phone = row.phone, r.cuisine = row.cuisine
            """,
            rows=restaurants,
        ).consume()
        session.run(
            """
            UNWIND $rows AS row
            MERGE (a:Address {id: row.id})
            SET a.label = row.label, a.value = row.value
            """,
            rows=addresses,
        ).consume()
        session.run(
            """
            UNWIND $rows AS row
            MERGE (c:City {id: row.id})
            SET c.label = row.label, c.value = row.value
            """,
            rows=cities,
        ).consume()
        session.run(
            """
            UNWIND $rows AS row
            MATCH (r:Restaurant {id: row.restaurant_id}), (a:Address {id: row.address_id})
            MERGE (r)-[:LOCATED_AT]->(a)
            """,
            rows=restaurant_address,
        ).consume()
        session.run(
            """
            UNWIND $rows AS row
            MATCH (a:Address {id: row.address_id}), (c:City {id: row.city_id})
            MERGE (a)-[:IN_CITY]->(c)
            """,
            rows=address_city,
        ).consume()

        counts = session.run(
            """
            MATCH (n)
            OPTIONAL MATCH ()-[e]->()
            RETURN count(DISTINCT n) AS nodes, count(DISTINCT e) AS edges
            """
        ).single()
        print(f"Imported graph: nodes={counts['nodes']} edges={counts['edges']}")

    driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import bundled Fodors-Zagats graph data into Neo4j.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/network"))
    parser.add_argument("--no-clear", action="store_true", help="Do not delete existing graph data before import.")
    args = parser.parse_args()
    import_graph(args.dataset_dir, clear=not args.no_clear)


if __name__ == "__main__":
    main()
