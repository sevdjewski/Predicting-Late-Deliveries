# Predicting Late Deliveries with Olist

This project investigates whether information available at purchase time can help identify orders at risk of late delivery. I compared logistic regression, gradient boosting, and a simple sorting rule using a fixed review budget of 10% of orders.

In the later evaluation period, sorting by shortest promised delivery time identified **279 of 674 late deliveries** while flagging **1,929 of 19,294 orders** for review. A logistic regression using only promised delivery time selected exactly the same orders.

## Dataset

The [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) contains anonymized orders from Brazilian online marketplaces. This analysis uses **96,470 delivered orders** with both an actual and an estimated delivery date. Of these, **6,534 (6.77%)** arrived late.

An order is labeled late when its actual delivery date falls after its estimated delivery date. Orders arriving on the estimated date count as on time.

## Method

- **Features.** I calculated the promised delivery time in days from the purchase date and the estimated delivery date. I also aggregated order items into three basket features: line-item count, total product value, and total shipping cost.
- **Data checks.** I joined the aggregated items to the orders by `order_id` and verified that all 96,470 orders were retained, with exactly one row per order and no unmatched orders.
- **Temporal evaluation.** I used a validation window starting on **13 March 2018** to compare model variants. I then refit the models on earlier orders with known delivery outcomes and evaluated them on orders placed from **26 May 2018** onward.
- **Leakage prevention.** The actual delivery date was used to define the target and determine whether the outcome was known at the training cutoff. The models used only the listed input features.
- **Metrics.** I used Average Precision to evaluate the risk ranking and precision and recall among the top 10% of ranked orders to measure performance at a fixed review budget.

## Results on Later Orders

The later period contained **19,294 delivered orders**, of which **674 (3.49%)** arrived late. Predicting every order as on time would achieve **96.51% accuracy** while identifying zero late deliveries.

| Approach | Average Precision | Late orders found | Precision at top 10% | Recall at top 10% |
| --- | ---: | ---: | ---: | ---: |
| Sort by shortest promised delivery time | — | **279** | **14.46%** | **41.39%** |
| Logistic regression, promised delivery time only | **0.1203** | **279** | **14.46%** | **41.39%** |
| Logistic regression with basket features | 0.062 | 102 | 5.29% | 15.13% |
| Histogram gradient boosting with basket features | 0.036 | 81 | 4.20% | 12.02% |

Each approach flagged **1,929 orders**. Precision is the share of flagged orders that arrived late. Recall is the share of all late orders that were flagged.

## What I Learned

Adding basket features improved validation performance. At the same review budget, the logistic regression with basket features found **138** late orders versus **47** for the single-feature model. Gradient boosting found **144**. That ranking reversed in the later period.

The later period had **1,260 orders** with a promised delivery window of **0–7 days**, a group absent from the earlier validation sample. The sorting rule flagged all 1,260, and **213** arrived late. Those orders accounted for about **76%** of the rule’s true positives.

For this dataset and review budget, the sorting rule delivered the same operational result as the single-feature model. The comparison also showed how a model that performs better in one period can perform worse in the next.

## Limitations

- The analysis covers delivered orders with both dates available. Canceled and undelivered orders are outside its scope.
- The validation sample includes orders delivered before 26 May 2018. This may exclude some slower deliveries near the end of the validation window.
- The later period was already inspected during the initial baseline analysis. Its results are a temporal comparison rather than an untouched final evaluation.
- The 10% review budget is an example operating constraint. The analysis measures how many late orders were identified. It does not measure whether an intervention would have prevented the delays.

## Run Locally

1. Download and extract the [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
2. Place `olist_orders_dataset.csv` and `olist_order_items_dataset.csv` next to `01_explore_orders.ipynb`.
3. Create the Conda environment:

   ```bash
   conda create -n olist python=3.11 "numpy<2" pandas scikit-learn ipykernel
   ```

4. Open `01_explore_orders.ipynb` in VS Code, select the `olist` kernel, and run all cells from top to bottom.

## Tools

Python · pandas · NumPy · scikit-learn · Jupyter
