"""
Auto-evaluation script for Academic Paper Assistant.
Automatically generates and runs evaluation questions.
"""

import json
from src.document_processor import get_vectorstore, get_collection_stats, get_retriever
from src.rag_chain import build_streaming_chain, format_docs_with_metadata


def generate_eval_questions(paper_names):
    """Auto-generate evaluation questions based on indexed papers."""
    questions = []
    
    # Generic questions that work for most papers
    base_questions = [
        ("What methodology was used?", "methodology"),
        ("What datasets were used?", "datasets"),
        ("What are the main findings?", "findings"),
        ("What are the limitations?", "limitations"),
        ("What is the main contribution?", "contribution"),
    ]
    
    # Single-paper questions
    for paper in paper_names[:3]:  # Test first 3 papers
        for q_text, category in base_questions[:3]:
            questions.append({
                "id": len(questions) + 1,
                "question": q_text,
                "expected_papers": [paper],
                "category": category
            })
    
    # Multi-paper questions
    if len(paper_names) >= 2:
        questions.extend([
            {
                "id": len(questions) + 1,
                "question": "Compare the approaches used across papers",
                "expected_papers": paper_names[:2],
                "category": "cross-paper"
            },
            {
                "id": len(questions) + 2,
                "question": "What datasets were used in these papers?",
                "expected_papers": paper_names[:3],
                "category": "cross-paper"
            }
        ])
    
    return questions


def evaluate_retrieval(vectorstore, question, expected_papers, k=8):
    """Check if expected papers appear in top-K chunks."""
    retriever = get_retriever(vectorstore)
    docs = retriever.invoke(question)
    
    retrieved = set(doc.metadata.get("source_file", "") for doc in docs[:k])
    expected = set(expected_papers)
    hits = len(expected.intersection(retrieved))
    recall = hits / len(expected) if expected else 0
    
    return {"recall@k": recall, "retrieved": list(retrieved), "hits": hits}


def evaluate_citations(answer):
    """Check if answer contains source references."""
    has_citation = "[Paper:" in answer or "[Source:" in answer or "Source:" in answer
    count = answer.count("[Paper:") + answer.count("[Source:") + answer.count("Source:")
    return {"has_citation": has_citation, "count": count}


def run_auto_evaluation():
    """Auto-generate questions and run evaluation."""
    print("\n" + "="*60)
    print("AUTO-EVALUATION FOR ACADEMIC PAPER ASSISTANT")
    print("="*60 + "\n")
    
    vectorstore = get_vectorstore()
    stats = get_collection_stats(vectorstore)
    
    if stats["indexed_papers"] == 0:
        print("No papers indexed. Please index papers first.")
        return
    
    print(f"Found {stats['indexed_papers']} indexed papers:")
    for i, paper in enumerate(stats['paper_names'], 1):
        print(f"  {i}. {paper}")
    
    # Auto-generate questions
    questions = generate_eval_questions(stats['paper_names'])
    print(f"\nGenerated {len(questions)} evaluation questions\n")
    
    chain, retriever = build_streaming_chain(vectorstore)
    
    results = []
    total_recall = 0
    total_citations = 0
    
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['question'][:60]}...")
        
        try:
            # Get answer
            answer = "".join(chain.stream(q["question"]))
            
            # Metrics
            ret = evaluate_retrieval(vectorstore, q["question"], q["expected_papers"])
            cit = evaluate_citations(answer)
            
            results.append({
                "question": q["question"],
                "expected_papers": q["expected_papers"],
                "recall@8": ret["recall@k"],
                "has_citation": cit["has_citation"],
                "citation_count": cit["count"]
            })
            
            total_recall += ret["recall@k"]
            total_citations += 1 if cit["has_citation"] else 0
            
            print(f"  Recall@8: {ret['recall@k']:.2f} | Citations: {'✓' if cit['has_citation'] else '✗'}")
        
        except Exception as e:
            print(f"  Error: {str(e)}")
            continue
    
    # Summary
    if results:
        summary = {
            "total_papers": stats["indexed_papers"],
            "paper_names": stats["paper_names"],
            "total_questions": len(results),
            "avg_recall@8": total_recall / len(results),
            "citation_rate": total_citations / len(results),
            "results": results
        }
        
        with open("evaluation_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Papers Indexed: {summary['total_papers']}")
        print(f"Questions Tested: {summary['total_questions']}")
        print(f"Average Recall@8: {summary['avg_recall@8']:.2%}")
        print(f"Citation Rate: {summary['citation_rate']:.2%}")
        print(f"\nDetailed results: evaluation_results.json")
        print("="*60 + "\n")
        
        return summary
    else:
        print("\nNo results generated.")
        return None


if __name__ == "__main__":
    run_auto_evaluation()