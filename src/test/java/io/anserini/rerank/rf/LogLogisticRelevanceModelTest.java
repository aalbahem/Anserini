package io.anserini.rerank.rf;

import io.anserini.rerank.RerankerContext;
import io.anserini.rerank.ScoredDocuments;
import io.anserini.rerank.rf.models.LogLogisticRelevanceModel;
import io.anserini.search.SearchArgs;
import io.anserini.util.FeatureVector;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.*;
import org.apache.lucene.queryparser.flexible.standard.StandardQueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.RAMDirectory;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.StringReader;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;

public class LogLogisticRelevanceModelTest {

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
  public void estimate() throws Exception {


    StandardQueryParser queryParser = new StandardQueryParser(analyzer);
    Query query = queryParser.parse(queryText, CONTENT_FIELD);
    IndexSearcher indexSearcher = new IndexSearcher(indexReader);

    TopDocs topDocs = indexSearcher.search(query, 10);
    SearchArgs searchArgs = new SearchArgs();
    searchArgs.arbitraryScoreTieBreak = true;

    double c = 0.5;
    double avgL = 13.0/3;
    double lambda = 2/3.0;

    context = new RerankerContext(indexSearcher, "1", query, "", queryText, null, null, searchArgs);
    FeedbackModelConfig config = new FeedbackModelConfig(10,10,0.5f,true,false,false,false,false);

    LogLogisticRelevanceModel model = new LogLogisticRelevanceModel(analyzer, CONTENT_FIELD, (float)c,config);



    model.setIndexSearcher(context.getIndexSearcher());
    FeatureVector llgmodel = model.estimate(queryText,
        ScoredDocuments.fromTopDocs(topDocs, indexSearcher),indexReader,false);


    float bananaFW = (float) (Math.log(((1 * Math.log(1 + c * (avgL/4.0))) + lambda)/lambda)
            + Math.log(((1 * Math.log(1 + c * (avgL/5.0))) + lambda)/lambda));

    float slugFW = (float) (Math.log(((1 * Math.log(1 + c * (avgL/4.0))) + lambda)/lambda)
            + Math.log(((1 * Math.log(1 + c * (avgL/5.0))) + lambda)/lambda));

    assertEquals(bananaFW/2.0f,llgmodel.getFeatureWeight("banana"),0.0);
    llgmodel = model.estimate(queryText, (String[])Arrays.copyOfRange(docs,0,2),
        ScoredDocuments.fromTopDocs(topDocs,indexSearcher).scores,indexReader,false);
    assertEquals(bananaFW/2.0f,llgmodel.getFeatureWeight("banana"),0.0);
  }
}