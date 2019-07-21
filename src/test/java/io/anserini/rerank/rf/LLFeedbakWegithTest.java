package io.anserini.rerank.rf.models;

import io.anserini.analysis.EnglishStemmingAnalyzer;
import io.anserini.rerank.RerankerContext;
import io.anserini.rerank.rf.FeedbackModel;
import io.anserini.rerank.rf.models.LLFeedbakWegith;
import io.anserini.rerank.rf.models.LogLogisticRelevanceModel;
import io.anserini.util.AnalyzerUtils;
import io.anserini.util.FeatureVector;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.*;
import org.apache.lucene.search.CollectionStatistics;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TermStatistics;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.RAMDirectory;
import org.apache.lucene.util.BytesRef;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.io.StringReader;
import java.util.HashMap;
import java.util.Set;

public class LLFeedbakWegithTest {


  RerankerContext context = null;
  Directory directory = null;
  String queryText = "banana slug";

  //this is taken from introduction to information retreical back
  String[] docs = {"banana slug Ariolimax columbianus","Santa Cruz mountains banana slug ","Santa Cruz Campus Mascot"};
  int[] rel = {0,1};
  String[] ids = {"1","2","3"};
  int[] noRel = {2};

  String ID_FIELD = "id";
  String CONTENT_FIELD = "contents";

  private IndexReader indexReader;
  private Analyzer analyzer;
  IndexSearcher indexSearcher;


  @Before
  public void setUp() throws Exception {
    directory = new RAMDirectory();
    analyzer = new EnglishAnalyzer();
    IndexWriterConfig config = new IndexWriterConfig(analyzer);

    IndexWriter indexWriter = new IndexWriter(directory, config);
    FieldType fieldType = new FieldType();
    fieldType.setTokenized(true);
    fieldType.setStored(true);
    fieldType.setStoreTermVectors(true);
    fieldType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);


    for (int i = 0; i < docs.length; i++){
      Document rD = new Document();

      rD.add(new TextField(ID_FIELD, new StringReader(ids[i])));

      Field field = new Field(CONTENT_FIELD, docs[i], fieldType);
      rD.add(field);
      indexWriter.addDocument(rD);
    }
    indexWriter.close();

    indexReader = DirectoryReader.open(directory);
    indexSearcher = new IndexSearcher(indexReader);


  }

  @After
  public void tearDown() throws Exception {

  }


  @Test
  public void t() throws IOException {

    FeatureVector vector = FeatureVector
        .fromTerms(AnalyzerUtils.tokenize(new EnglishStemmingAnalyzer(),docs[0]));


    HashMap<String,Float> FW = new HashMap<>();
    CollectionStatistics collectionStatistics =  indexSearcher.collectionStatistics(CONTENT_FIELD);

    float c = 0.5f;
    double avgL = 13/3.0;
    double tf = 1.0f;
    TermStates termContext = new TermStates(indexReader.getContext());

    String term = "banana";
    tf = 1.0f;
    TermStatistics termStats = indexSearcher.termStatistics(new Term(CONTENT_FIELD,new BytesRef(term)),termContext);
    LLFeedbakWegith llFeedbakWegith = new LLFeedbakWegith(0.5F,collectionStatistics,termStats);

    Assert.assertEquals(tf * Math.log(1 + c * (avgL/4.0)),llFeedbakWegith.t(tf,4.0),0.0);
    Assert.assertEquals(tf * Math.log(1 + c * (avgL/5.0)),llFeedbakWegith.t(tf,5.0),0.0);
    Assert.assertEquals(0 * Math.log(1 + c * (avgL/4.0)),llFeedbakWegith.t(0,4.0),0.0);
  }

  @Test
  public void computeFW1() throws IOException {
    FeatureVector vector = FeatureVector
        .fromTerms(AnalyzerUtils.tokenize(new EnglishStemmingAnalyzer(),docs[0]));


    HashMap<String,Float> FW = new HashMap<>();
    CollectionStatistics collectionStatistics =  indexSearcher.collectionStatistics(CONTENT_FIELD);

    float c = 0.5f;
    double avgL = 13/3.0;
    double tf = 1.0f;
    TermStates termContext = TermStates.build(indexReader.getContext(),new Term(CONTENT_FIELD,new BytesRef("banana")),true);

    String term = "banana";
    tf = 1.0f;
    double lambda = 2.0/3.0;
    TermStatistics termStats = indexSearcher.termStatistics(new Term(CONTENT_FIELD,new BytesRef(term)),termContext);

    LLFeedbakWegith llFeedbakWegith = new LLFeedbakWegith(0.5F,collectionStatistics,termStats);

    Assert.assertEquals(Math.log(((tf * Math.log(1 + c * (avgL/4.0))) + lambda)/lambda),llFeedbakWegith.computeFW(term,1.0,4.0),0.0);
    Assert.assertEquals(Math.log(((tf * Math.log(1 + c * (avgL/5.0))) + lambda)/lambda),llFeedbakWegith.computeFW(term,1.0,5.0),0.0);

  }



  public void computeFW() throws IOException {

    FeatureVector[] vectors = new FeatureVector[docs.length];

    for (int i =0; i < vectors.length; i++){
      vectors[i] = FeatureVector
          .fromTerms(AnalyzerUtils.tokenize(new EnglishStemmingAnalyzer(),docs[i]));
    }

    float[] norms = {4.0f,5.0f,4.0f};

    Set<String> Vocab = FeedbackModel.getVocab(vectors);

    HashMap<String,Double> FW = new HashMap<>();
    CollectionStatistics collectionStatistics =  indexSearcher.collectionStatistics(CONTENT_FIELD);
    for (String term : Vocab) {
      TermStates termContext =TermStates.build(indexReader.getContext(),new Term(CONTENT_FIELD,term),true);
      LLFeedbakWegith llFeedbakWegith = new LLFeedbakWegith(0.5F,collectionStatistics,indexSearcher.termStatistics(new Term(CONTENT_FIELD,new BytesRef(term)),termContext));

      FW.put(term,llFeedbakWegith.computeFW(term,norms,vectors));
    }
    System.out.println(FW.keySet());


    String[] expectedVocab =  {"banana", "mountains", "columbianus", "mascot", "campus", "cruz", "slug", "ariolimax", "santa"};

    double c = 0.5;
    double avgL = 13.0/3;
    double lambda = 2/3.0;
    double bananaFW =   Math.log(((1 * Math.log(1 + c * (avgL/4.0))) + lambda)/lambda)
                      + Math.log(((1 * Math.log(1 + c * (avgL/5.0))) + lambda)/lambda)
                      + Math.log(((0 * Math.log(1 + c * (avgL/4.0))) + lambda)/lambda);
    Assert.assertEquals(bananaFW/3,(double)FW.get("banana"),0.0);
  }

}