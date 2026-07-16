from __future__ import annotations

import argparse
import csv
from pathlib import Path

from config import neo4j_config
from dataset_config import DATASET_CONFIGS


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


def chunks(items: list[dict], size: int = 5000):
    for index in range(0, len(items), size):
        yield items[index:index + size]


def run_batches(session, query: str, rows_to_write: list[dict]) -> None:
    for batch in chunks(rows_to_write):
        session.run(query, rows=batch).consume()


def graph_counts(session) -> dict:
    nodes = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
    edges = session.run("MATCH ()-[e]->() RETURN count(e) AS count").single()["count"]
    return {"nodes": nodes, "edges": edges}


def import_fodors_zagats_graph(dataset_dir: Path = DATASET_CONFIGS["fz"].network_dir, clear: bool = True) -> None:
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

        run_batches(
            session,
            """
            UNWIND $rows AS row
            MERGE (r:Restaurant {id: row.id})
            SET r.label = row.label, r.name = row.name, r.phone = row.phone, r.cuisine = row.cuisine
            """,
            restaurants,
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MERGE (a:Address {id: row.id})
            SET a.label = row.label, a.value = row.value
            """,
            addresses,
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MERGE (c:City {id: row.id})
            SET c.label = row.label, c.value = row.value
            """,
            cities,
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MATCH (r:Restaurant {id: row.restaurant_id}), (a:Address {id: row.address_id})
            MERGE (r)-[:LOCATED_AT]->(a)
            """,
            restaurant_address,
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MATCH (a:Address {id: row.address_id}), (c:City {id: row.city_id})
            MERGE (a)-[:IN_CITY]->(c)
            """,
            address_city,
        )

        counts = graph_counts(session)
        print(f"Imported graph: nodes={counts['nodes']} edges={counts['edges']}")

    driver.close()


def import_dblp_acm_graph(dataset_dir: Path = DATASET_CONFIGS["dblp_acm"].network_dir, clear: bool = True) -> None:
    papers: dict[str, dict] = {}
    for row in rows(dataset_dir / "paper.txt"):
        if len(row) >= 4:
            year = row[3].replace(".0", "")
            papers[row[0]] = {
                "id": row[0],
                "label": row[1],
                "name": row[2],
                "title": row[2],
                "phone": year,
                "year": year,
                "cuisine": "",
                "authors": "",
                "venue": "",
            }

    authors = [{"id": row[0], "label": row[1], "value": row[2]} for row in rows(dataset_dir / "author.txt") if len(row) >= 3]
    venues = [{"id": row[0], "label": row[1], "value": row[2]} for row in rows(dataset_dir / "venue.txt") if len(row) >= 3]
    author_by_id = {row["id"]: row["value"] for row in authors}
    venue_by_id = {row["id"]: row["value"] for row in venues}

    paper_authors: dict[str, list[str]] = {}
    author_paper = []
    for row in rows(dataset_dir / "author_paper.txt"):
        if len(row) >= 2 and row[1] in papers:
            author_paper.append({"author_id": row[0], "paper_id": row[1]})
            if row[0] in author_by_id:
                paper_authors.setdefault(row[1], []).append(author_by_id[row[0]])

    paper_venue = []
    for row in rows(dataset_dir / "paper_venue.txt"):
        if len(row) >= 2 and row[0] in papers:
            paper_venue.append({"paper_id": row[0], "venue_id": row[1]})
            if row[1] in venue_by_id:
                papers[row[0]]["venue"] = venue_by_id[row[1]]
                papers[row[0]]["cuisine"] = venue_by_id[row[1]]

    for paper_id, values in paper_authors.items():
        papers[paper_id]["authors"] = "; ".join(values[:8])

    driver = get_driver()
    with driver.session() as session:
        if clear:
            session.run("MATCH (n) DETACH DELETE n").consume()

        session.run("CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE").consume()
        session.run("CREATE CONSTRAINT author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE").consume()
        session.run("CREATE CONSTRAINT venue_id IF NOT EXISTS FOR (v:Venue) REQUIRE v.id IS UNIQUE").consume()

        run_batches(
            session,
            """
            UNWIND $rows AS row
            MERGE (p:Paper {id: row.id})
            SET p.label = row.label, p.name = row.name, p.title = row.title,
                p.phone = row.phone, p.year = row.year, p.cuisine = row.cuisine,
                p.authors = row.authors, p.venue = row.venue
            """,
            list(papers.values()),
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MERGE (a:Author {id: row.id})
            SET a.label = row.label, a.value = row.value
            """,
            authors,
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MERGE (v:Venue {id: row.id})
            SET v.label = row.label, v.value = row.value
            """,
            venues,
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MATCH (p:Paper {id: row.paper_id}), (a:Author {id: row.author_id})
            MERGE (p)-[:WRITTEN_BY]->(a)
            """,
            author_paper,
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MATCH (p:Paper {id: row.paper_id}), (v:Venue {id: row.venue_id})
            MERGE (p)-[:PUBLISHED_IN]->(v)
            """,
            paper_venue,
        )

        counts = graph_counts(session)
        print(f"Imported graph: nodes={counts['nodes']} edges={counts['edges']}")

    driver.close()


def import_amazon_google_graph(dataset_dir: Path = DATASET_CONFIGS["amazon_google"].network_dir, clear: bool = True) -> None:
    software: dict[str, dict] = {}
    for row in rows(dataset_dir / "software.txt"):
        if len(row) >= 3:
            price = row[3] if len(row) >= 4 else ""
            software[row[0]] = {
                "id": row[0],
                "label": row[1],
                "name": row[2],
                "title": row[2],
                "phone": price,
                "price": price,
                "cuisine": "",
                "manufacturer": "",
            }

    manufacturers = [
        {"id": row[0], "label": row[0], "value": row[1]}
        for row in rows(dataset_dir / "manufacturer.txt")
        if len(row) >= 2
    ]
    manufacturer_by_id = {row["id"]: row["value"] for row in manufacturers}

    software_manufacturer = []
    for row in rows(dataset_dir / "software_manufacturer.txt"):
        if len(row) >= 2 and row[0] in software:
            software_manufacturer.append({"software_id": row[0], "manufacturer_id": row[1]})
            if row[1] in manufacturer_by_id:
                software[row[0]]["manufacturer"] = manufacturer_by_id[row[1]]
                software[row[0]]["cuisine"] = manufacturer_by_id[row[1]]

    driver = get_driver()
    with driver.session() as session:
        if clear:
            session.run("MATCH (n) DETACH DELETE n").consume()

        session.run("CREATE CONSTRAINT software_id IF NOT EXISTS FOR (s:Software) REQUIRE s.id IS UNIQUE").consume()
        session.run("CREATE CONSTRAINT manufacturer_id IF NOT EXISTS FOR (m:Manufacturer) REQUIRE m.id IS UNIQUE").consume()

        run_batches(
            session,
            """
            UNWIND $rows AS row
            MERGE (s:Software {id: row.id})
            SET s.label = row.label, s.name = row.name, s.title = row.title,
                s.phone = row.phone, s.price = row.price, s.cuisine = row.cuisine,
                s.manufacturer = row.manufacturer
            """,
            list(software.values()),
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MERGE (m:Manufacturer {id: row.id})
            SET m.label = row.label, m.value = row.value
            """,
            manufacturers,
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MATCH (s:Software {id: row.software_id}), (m:Manufacturer {id: row.manufacturer_id})
            MERGE (s)-[:MADE_BY]->(m)
            """,
            software_manufacturer,
        )

        counts = graph_counts(session)
        print(f"Imported graph: nodes={counts['nodes']} edges={counts['edges']}")

    driver.close()


def import_citeseer_graph(dataset_dir: Path = DATASET_CONFIGS["citeseer"].network_dir, clear: bool = True) -> None:
    authors = [
        {
            "id": row[0],
            "label": row[1],
            "name": row[2],
            "title": row[2],
            "phone": "",
            "cuisine": "",
            "paper": "",
        }
        for row in rows(dataset_dir / "author.txt")
        if len(row) >= 3
    ]
    author_by_id = {row["id"]: row for row in authors}
    papers = [
        {"id": row[0], "label": row[1], "value": row[2] if len(row) >= 3 else row[1], "title": row[2] if len(row) >= 3 else row[1]}
        for row in rows(dataset_dir / "paper.txt")
        if len(row) >= 2
    ]
    paper_by_id = {row["id"]: row["value"] for row in papers}
    author_paper = []
    for row in rows(dataset_dir / "author-paper.txt"):
        if len(row) >= 2 and row[0] in author_by_id:
            author_paper.append({"author_id": row[0], "paper_id": row[1]})
            author_by_id[row[0]]["paper"] = paper_by_id.get(row[1], "")
            author_by_id[row[0]]["cuisine"] = paper_by_id.get(row[1], "")

    driver = get_driver()
    with driver.session() as session:
        if clear:
            session.run("MATCH (n) DETACH DELETE n").consume()

        session.run("CREATE CONSTRAINT citeseer_author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE").consume()
        session.run("CREATE CONSTRAINT citeseer_paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE").consume()

        run_batches(
            session,
            """
            UNWIND $rows AS row
            MERGE (a:Author {id: row.id})
            SET a.label = row.label, a.name = row.name, a.title = row.title,
                a.phone = row.phone, a.cuisine = row.cuisine, a.paper = row.paper
            """,
            authors,
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MERGE (p:Paper {id: row.id})
            SET p.label = row.label, p.value = row.value, p.title = row.title
            """,
            papers,
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MATCH (a:Author {id: row.author_id}), (p:Paper {id: row.paper_id})
            MERGE (a)-[:AUTHORED]->(p)
            """,
            author_paper,
        )

        counts = graph_counts(session)
        print(f"Imported graph: nodes={counts['nodes']} edges={counts['edges']}")

    driver.close()


def import_graph(dataset_key: str = "fz", dataset_dir: Path | None = None, clear: bool = True) -> None:
    config = DATASET_CONFIGS[dataset_key]
    source_dir = dataset_dir or config.network_dir
    if config.graph_kind == "fodors_zagats":
        import_fodors_zagats_graph(source_dir, clear=clear)
    elif config.graph_kind == "paper_author_venue":
        import_dblp_acm_graph(source_dir, clear=clear)
    elif config.graph_kind == "software_manufacturer":
        import_amazon_google_graph(source_dir, clear=clear)
    elif config.graph_kind == "author_paper":
        import_citeseer_graph(source_dir, clear=clear)
    else:
        raise ValueError(f"Unsupported graph kind: {config.graph_kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import bundled graph data into Neo4j.")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default="fz")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--no-clear", action="store_true", help="Do not delete existing graph data before import.")
    args = parser.parse_args()
    import_graph(args.dataset, args.dataset_dir, clear=not args.no_clear)


if __name__ == "__main__":
    main()
