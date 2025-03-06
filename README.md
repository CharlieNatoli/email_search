# Keyword-based RAG for Email Marketing

During my time at Klaviyo, I learned a lot about the daily workflows of ECommerce email marketers. One common workflow is to gather creative inspiration from other marketing emails, often through sites like [reallygoodemails.com](https://reallygoodemails.com]) or [milled.com](milled.com).
    
For this (quick) project, I wanted to test if I could build an image search tool specific to marketing terms. I created a search method I've named Keyword RAG. This system first uses a multimodal LLM and chain of thought reasoning to create domain specific keywords, then uses these as a basis for the search ranking.

## Project contents 

* jupyter notebooks: https://github.com/CharlieNatoli/email_search/tree/master/notebooks
* code for creating and query indices: https://github.com/CharlieNatoli/email_search/tree/master/utlities
 
## Dataset and methology

Dataset: ~300 example marketing emails downloaded from blog posts.

Image search indices: 
1. **Keyword RAG** - described above. Uses Claude as a multimodal LLM. Prompt to Claude focuses on specific marketing email types. 
2. **Baseline** - built off of OpenAI's CLIP.

Testing: compared results for queries related to common marketing email types (such as those from [really good emails](https://reallygoodemails.com/categories])).

## Results / Conclusions

See [Notebook](https://github.com/CharlieNatoli/email_search/blob/master/notebooks/Results.ipynb).

Keyword RAG generates more relevant results that CLIP-based index when asked about marketing-specific concepts that aren't immediately visible in the email (eg. "Abandoned Shopping Cart"). In contrast, both indices perform the same if the concept that is searched for is more explicitly  mentioned or visible in the email (eg "Clearance Sale") 

Overall, this system could work well for a smaller, curated content library the need to search using very domain-specific terminology. 

However, creating tags for each example image might not scale well to larger datasets due to cost. Also, adding domain specificity sacrifices much of the generality of a more standard image search. 



## Extensions

- build more formal evaluation suite
- add in reranker and/or filter irrelevant results. 
- try other multimodal embeddings other than CLIP.
- marketing-specific fine-tuning. 
