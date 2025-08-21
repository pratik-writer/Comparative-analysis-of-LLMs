import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os
from datetime import datetime

class ImageCaptioningEvaluator:
    def __init__(self, csv_file_path):
        """
        Initialize the evaluator with CSV file path
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.results = None
        self.models = ['GPT ', 'BLIP-2 ', 'ClipCap ', 'Claude ']
        self.reference_columns = ['Caption 1', 'Caption 2', 'Caption 3', 'Caption 4', 'Caption 5']
        
        # Create output directory
        self.output_dir = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """Load and preprocess the CSV data"""
        print("Loading data...")
        self.data = pd.read_csv(self.csv_file_path)
        
        # Clean whitespace
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                self.data[col] = self.data[col].astype(str).str.strip()
        
        print(f"Loaded {len(self.data)} images")
        return self.data
    
    def calculate_bleu4(self, candidate, references):
        """
        Calculate BLEU-4 score
        """
        def get_ngrams(text, n):
            tokens = text.lower().split()
            if len(tokens) < n:
                return []
            return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        candidate_tokens = candidate.lower().split()
        if len(candidate_tokens) == 0:
            return 0.0
        
        # Calculate precision for n-grams 1 to 4
        precisions = []
        
        for n in range(1, 5):
            candidate_ngrams = get_ngrams(candidate, n)
            if not candidate_ngrams:
                precisions.append(0.0)
                continue
                
            reference_ngrams = []
            for ref in references:
                reference_ngrams.extend(get_ngrams(ref, n))
            
            if not reference_ngrams:
                precisions.append(0.0)
                continue
            
            candidate_counter = Counter(candidate_ngrams)
            reference_counter = Counter(reference_ngrams)
            
            clipped_counts = sum(min(candidate_counter[ngram], reference_counter[ngram]) 
                               for ngram in candidate_counter)
            
            precision = clipped_counts / len(candidate_ngrams) if candidate_ngrams else 0
            precisions.append(precision)
        
        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            geometric_mean = np.exp(np.mean(np.log(precisions)))
        else:
            geometric_mean = 0.0
        
        # Brevity penalty
        ref_lengths = [len(ref.split()) for ref in references]
        closest_ref_len = min(ref_lengths, key=lambda x: abs(x - len(candidate_tokens)))
        
        if len(candidate_tokens) > closest_ref_len:
            bp = 1.0
        else:
            bp = np.exp(1 - closest_ref_len / len(candidate_tokens)) if len(candidate_tokens) > 0 else 0
        
        return geometric_mean * bp
    
    def calculate_rouge_l(self, candidate, references):
        """
        Calculate ROUGE-L score
        """
        def lcs_length(seq1, seq2):
            m, n = len(seq1), len(seq2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if seq1[i-1] == seq2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        candidate_tokens = candidate.lower().split()
        max_f1 = 0.0
        
        for reference in references:
            reference_tokens = reference.lower().split()
            
            if not candidate_tokens or not reference_tokens:
                continue
            
            lcs_len = lcs_length(candidate_tokens, reference_tokens)
            
            if lcs_len == 0:
                continue
            
            precision = lcs_len / len(candidate_tokens)
            recall = lcs_len / len(reference_tokens)
            
            if precision + recall > 0:
                f1 = (2 * precision * recall) / (precision + recall)
                max_f1 = max(max_f1, f1)
        
        return max_f1
    
    def calculate_cider(self, candidate, references):
        """
        Calculate CIDEr score (simplified version)
        """
        def get_ngrams_with_counts(text, n):
            tokens = text.lower().split()
            ngrams = []
            for i in range(len(tokens) - n + 1):
                ngrams.append(' '.join(tokens[i:i+n]))
            return Counter(ngrams)
        
        candidate_tokens = candidate.lower().split()
        if not candidate_tokens:
            return 0.0
        
        scores = []
        
        for n in range(1, 5):  # 1 to 4-grams
            candidate_ngrams = get_ngrams_with_counts(candidate, n)
            
            if not candidate_ngrams:
                scores.append(0.0)
                continue
            
            reference_ngram_counts = []
            for ref in references:
                ref_ngrams = get_ngrams_with_counts(ref, n)
                reference_ngram_counts.append(ref_ngrams)
            
            if not any(reference_ngram_counts):
                scores.append(0.0)
                continue
            
            # Calculate TF-IDF like scoring
            score = 0.0
            for ngram, count in candidate_ngrams.items():
                # Count how many references contain this n-gram
                ref_count = sum(1 for ref_ngrams in reference_ngram_counts if ngram in ref_ngrams)
                
                if ref_count > 0:
                    # Simple consensus scoring
                    consensus_weight = ref_count / len(references)
                    score += count * consensus_weight
            
            # Normalize by candidate length
            total_candidate_ngrams = sum(candidate_ngrams.values())
            if total_candidate_ngrams > 0:
                score = score / total_candidate_ngrams
            
            scores.append(score)
        
        return np.mean(scores)
    
    def calculate_bertscore(self, candidate, references):
        """
        Calculate simplified semantic similarity (BERTScore approximation)
        Using word overlap with semantic weighting
        """
        candidate_words = set(candidate.lower().split())
        if not candidate_words:
            return 0.0
        
        max_similarity = 0.0
        
        for reference in references:
            reference_words = set(reference.lower().split())
            
            if not reference_words:
                continue
            
            # Jaccard similarity with length normalization
            intersection = candidate_words.intersection(reference_words)
            
            if not intersection:
                continue
            
            # Precision and recall
            precision = len(intersection) / len(candidate_words)
            recall = len(intersection) / len(reference_words)
            
            # F1-like score
            if precision + recall > 0:
                similarity = (2 * precision * recall) / (precision + recall)
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def calculate_vocabulary_diversity(self, captions):
        """Calculate vocabulary diversity for a set of captions"""
        all_words = []
        for caption in captions:
            words = caption.lower().split()
            all_words.extend(words)
        
        if not all_words:
            return 0.0
        
        unique_words = set(all_words)
        return len(unique_words) / len(all_words)
    
    def calculate_average_length(self, captions):
        """Calculate average caption length"""
        lengths = [len(caption.split()) for caption in captions]
        return np.mean(lengths) if lengths else 0.0
    
    def calculate_repetition_rate(self, captions):
        """Calculate repetition rate in captions"""
        total_repetitions = 0
        total_bigrams = 0
        
        for caption in captions:
            words = caption.lower().split()
            if len(words) < 2:
                continue
            
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            
            # Count repetitions (bigrams that appear more than once)
            repetitions = sum(count - 1 for count in bigram_counts.values() if count > 1)
            
            total_repetitions += repetitions
            total_bigrams += len(bigrams)
        
        return total_repetitions / total_bigrams if total_bigrams > 0 else 0.0
    
    def evaluate_all_models(self):
        """
        Evaluate all models on all metrics
        """
        if self.data is None:
            self.load_data()
        
        print("Starting evaluation...")
        
        results = []
        model_captions = {model: [] for model in self.models}
        
        # Process each image
        for idx, row in self.data.iterrows():
            image_id = row['Image ID']
            
            # Get reference captions
            references = []
            for col in self.reference_columns:
                if pd.notna(row[col]) and str(row[col]).strip():
                    references.append(str(row[col]).strip())
            
            if not references:
                continue
            
            # Evaluate each model
            for model in self.models:
                if pd.notna(row[model]) and str(row[model]).strip():
                    candidate = str(row[model]).strip()
                    model_captions[model].append(candidate)
                    
                    # Calculate all metrics
                    bleu4 = self.calculate_bleu4(candidate, references)
                    rouge_l = self.calculate_rouge_l(candidate, references)
                    cider = self.calculate_cider(candidate, references)
                    bertscore = self.calculate_bertscore(candidate, references)
                    
                    results.append({
                        'Image_ID': image_id,
                        'Model': model.strip(),
                        'Caption': candidate,
                        'BLEU4': bleu4,
                        'ROUGE_L': rouge_l,
                        'CIDEr': cider,
                        'BERTScore': bertscore,
                        'Caption_Length': len(candidate.split())
                    })
            
            if (idx + 1) % 5 == 0:
                print(f"Processed {idx + 1}/{len(self.data)} images")
        
        # Calculate quality metrics for each model
        quality_metrics = {}
        for model in self.models:
            model_name = model.strip()
            captions = model_captions[model]
            
            if captions:
                quality_metrics[model_name] = {
                    'Vocabulary_Diversity': self.calculate_vocabulary_diversity(captions),
                    'Average_Length': self.calculate_average_length(captions),
                    'Repetition_Rate': self.calculate_repetition_rate(captions)
                }
        
        self.results = pd.DataFrame(results)
        self.quality_metrics = quality_metrics
        
        print("Evaluation completed!")
        return self.results, quality_metrics
    
    def generate_summary_statistics(self):
        """Generate summary statistics for each model"""
        if self.results is None:
            raise ValueError("No results available. Run evaluation first.")
        
        summary = self.results.groupby('Model').agg({
            'BLEU4': ['mean', 'std'],
            'ROUGE_L': ['mean', 'std'],
            'CIDEr': ['mean', 'std'],
            'BERTScore': ['mean', 'std'],
            'Caption_Length': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
        
        # Add quality metrics
        for model in summary.index:
            if model in self.quality_metrics:
                summary.loc[model, 'Vocabulary_Diversity'] = self.quality_metrics[model]['Vocabulary_Diversity']
                summary.loc[model, 'Average_Length_Quality'] = self.quality_metrics[model]['Average_Length']
                summary.loc[model, 'Repetition_Rate'] = self.quality_metrics[model]['Repetition_Rate']
        
        return summary
    
    def create_visualizations(self):
        """Create and save visualization charts"""
        if self.results is None:
            raise ValueError("No results available. Run evaluation first.")
        
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Performance comparison chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Image Captioning Models - Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['BLEU4', 'ROUGE_L', 'CIDEr', 'BERTScore']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            # Box plot for each metric
            model_data = []
            model_labels = []
            
            for model in self.models:
                model_name = model.strip()
                data = self.results[self.results['Model'] == model_name][metric].values
                if len(data) > 0:
                    model_data.append(data)
                    model_labels.append(model_name)
            
            if model_data:
                bp = ax.boxplot(model_data, labels=model_labels, patch_artist=True)
                
                # Color the boxes
                colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                
                ax.set_title(f'{metric} Distribution', fontweight='bold')
                ax.set_ylabel('Score')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Quality metrics comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Quality Metrics Comparison', fontsize=16, fontweight='bold')
        
        quality_data = []
        models_list = []
        
        for model in self.models:
            model_name = model.strip()
            if model_name in self.quality_metrics:
                quality_data.append([
                    self.quality_metrics[model_name]['Vocabulary_Diversity'],
                    self.quality_metrics[model_name]['Average_Length'],
                    self.quality_metrics[model_name]['Repetition_Rate']
                ])
                models_list.append(model_name)
        
        if quality_data:
            quality_df = pd.DataFrame(quality_data, 
                                    columns=['Vocabulary Diversity', 'Average Length', 'Repetition Rate'],
                                    index=models_list)
            
            # Plot each quality metric
            for i, metric in enumerate(['Vocabulary Diversity', 'Average Length', 'Repetition Rate']):
                ax = axes[i]
                bars = ax.bar(models_list, quality_df[metric], 
                            color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'][:len(models_list)])
                ax.set_title(metric, fontweight='bold')
                ax.set_ylabel('Value')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, quality_df[metric]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/quality_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Radar chart for overall performance
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Calculate average scores for radar chart
        avg_scores = self.results.groupby('Model')[['BLEU4', 'ROUGE_L', 'CIDEr', 'BERTScore']].mean()
        
        # Normalize scores to 0-1 for better visualization
        normalized_scores = avg_scores.div(avg_scores.max())
        
        angles = np.linspace(0, 2 * np.pi, len(normalized_scores.columns), endpoint=False).tolist()
        angles += angles[:1]  
        
        colors = ['blue', 'green', 'red', 'orange']
        
        for i, (model, scores) in enumerate(normalized_scores.iterrows()):
            values = scores.tolist()
            values += values[:1]  
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(normalized_scores.columns)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Performance Radar Chart', fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {self.output_dir}/")
    
    def save_results(self):
        """Save all results to CSV files"""
        if self.results is None:
            raise ValueError("No results available. Run evaluation first.")
        
        print("Saving results...")
        
        # 1. Detailed results (one row per image-model combination)
        self.results.to_csv(f'{self.output_dir}/detailed_results.csv', index=False)
        
        # 2. Summary statistics
        summary = self.generate_summary_statistics()
        summary.to_csv(f'{self.output_dir}/summary_statistics.csv')
        
        # 3. Quality metrics
        quality_df = pd.DataFrame(self.quality_metrics).T
        quality_df.to_csv(f'{self.output_dir}/quality_metrics.csv')
        
        # 4. Model rankings
        avg_scores = self.results.groupby('Model')[['BLEU4', 'ROUGE_L', 'CIDEr', 'BERTScore']].mean()
        
        rankings = {}
        for metric in ['BLEU4', 'ROUGE_L', 'CIDEr', 'BERTScore']:
            rankings[f'{metric}_Rank'] = avg_scores[metric].rank(method='dense', ascending=False).astype(int)
        
        rankings_df = pd.DataFrame(rankings, index=avg_scores.index)
        rankings_df['Overall_Rank'] = rankings_df.mean(axis=1).rank(method='dense', ascending=True).astype(int)
        rankings_df.to_csv(f'{self.output_dir}/model_rankings.csv')
        
        print(f"Results saved to {self.output_dir}/")
        return self.output_dir
    
    def run_complete_evaluation(self):
        """Run the complete evaluation pipeline"""
        print("="*60)
        print("IMAGE CAPTIONING MODELS EVALUATION")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Run evaluation
        self.evaluate_all_models()
        
        # Generate summary
        summary = self.generate_summary_statistics()
        print("\nSUMMARY STATISTICS:")
        print("-" * 40)
        print(summary.round(4))
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        output_dir = self.save_results()
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETED!")
        print(f"Results saved to: {output_dir}")
        print("Files created:")
        print(" detailed_results.csv (per-image scores)")
        print(" summary_statistics.csv (model averages)")
        print(" quality_metrics.csv (diversity, length, repetition)")
        print(" model_rankings.csv (performance rankings)")
        print(" performance_comparison.png")
        print(" quality_metrics.png")
        print(" radar_chart.png")
        print("="*60)
        
        return output_dir


if __name__ == "__main__":
    
    csv_file = "ready_final_file.csv"
    
    evaluator = ImageCaptioningEvaluator(csv_file)

    output_directory = evaluator.run_complete_evaluation()
