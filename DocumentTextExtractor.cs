using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;

namespace DocumentTextExtractor
{
    public class DocumentTextExtractor
    {
        /// <summary>
        /// Extracts all text content from a Word document
        /// </summary>
        /// <param name="filePath">Path to the .docx file</param>
        /// <returns>Complete text content of the document</returns>
        public static string ExtractAllText(string filePath)
        {
            try
            {
                using (var wordDocument = WordprocessingDocument.Open(filePath, false))
                {
                    var mainPart = wordDocument.MainDocumentPart;
                    if (mainPart?.Document?.Body == null)
                        return string.Empty;

                    var textBuilder = new StringBuilder();

                    // Extract main document text
                    var mainText = ExtractTextFromBody(mainPart.Document.Body);
                    if (!string.IsNullOrEmpty(mainText))
                        textBuilder.AppendLine(mainText);

                    // Extract headers
                    var headerText = ExtractHeadersText(mainPart);
                    if (!string.IsNullOrEmpty(headerText))
                    {
                        textBuilder.AppendLine("=== HEADERS ===");
                        textBuilder.AppendLine(headerText);
                    }

                    // Extract footers
                    var footerText = ExtractFootersText(mainPart);
                    if (!string.IsNullOrEmpty(footerText))
                    {
                        textBuilder.AppendLine("=== FOOTERS ===");
                        textBuilder.AppendLine(footerText);
                    }

                    return textBuilder.ToString().Trim();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading Word document: {ex.Message}");
                return string.Empty;
            }
        }

        /// <summary>
        /// Extracts text content from the main document body
        /// </summary>
        /// <param name="filePath">Path to the .docx file</param>
        /// <returns>Main document text content</returns>
        public static string ExtractMainText(string filePath)
        {
            try
            {
                using (var wordDocument = WordprocessingDocument.Open(filePath, false))
                {
                    var mainPart = wordDocument.MainDocumentPart;
                    if (mainPart?.Document?.Body == null)
                        return string.Empty;

                    return ExtractTextFromBody(mainPart.Document.Body);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading main document text: {ex.Message}");
                return string.Empty;
            }
        }

        /// <summary>
        /// Extracts text and organizes it by document sections
        /// </summary>
        /// <param name="filePath">Path to the .docx file</param>
        /// <returns>Document content organized by sections</returns>
        public static DocumentContent ExtractStructuredContent(string filePath)
        {
            var docContent = new DocumentContent();

            try
            {
                using (var wordDocument = WordprocessingDocument.Open(filePath, false))
                {
                    var mainPart = wordDocument.MainDocumentPart;
                    if (mainPart?.Document?.Body == null)
                        return docContent;

                    // Extract paragraphs
                    docContent.Paragraphs = ExtractParagraphs(mainPart.Document.Body);

                    // Extract tables
                    docContent.Tables = ExtractTables(mainPart.Document.Body);

                    // Extract headers
                    docContent.Headers = ExtractHeadersList(mainPart);

                    // Extract footers
                    docContent.Footers = ExtractFootersList(mainPart);

                    // Extract document properties
                    docContent.Properties = ExtractDocumentProperties(wordDocument);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error extracting structured content: {ex.Message}");
            }

            return docContent;
        }

        /// <summary>
        /// Extracts text from document body
        /// </summary>
        private static string ExtractTextFromBody(Body body)
        {
            var textBuilder = new StringBuilder();

            foreach (var element in body.Elements())
            {
                switch (element)
                {
                    case Paragraph paragraph:
                        var paragraphText = ExtractTextFromParagraph(paragraph);
                        if (!string.IsNullOrWhiteSpace(paragraphText))
                            textBuilder.AppendLine(paragraphText);
                        break;

                    case Table table:
                        var tableText = ExtractTextFromTable(table);
                        if (!string.IsNullOrWhiteSpace(tableText))
                            textBuilder.AppendLine(tableText);
                        break;
                }
            }

            return textBuilder.ToString().Trim();
        }

        /// <summary>
        /// Extracts text from a paragraph
        /// </summary>
        private static string ExtractTextFromParagraph(Paragraph paragraph)
        {
            var textBuilder = new StringBuilder();

            foreach (var run in paragraph.Elements<Run>())
            {
                foreach (var text in run.Elements<Text>())
                {
                    textBuilder.Append(text.Text);
                }
            }

            return textBuilder.ToString().Trim();
        }

        /// <summary>
        /// Extracts text from a table
        /// </summary>
        private static string ExtractTextFromTable(Table table)
        {
            var textBuilder = new StringBuilder();
            textBuilder.AppendLine("[TABLE]");

            foreach (var row in table.Elements<TableRow>())
            {
                var cellTexts = new List<string>();
                foreach (var cell in row.Elements<TableCell>())
                {
                    var cellText = string.Join(" ", cell.Elements<Paragraph>()
                        .Select(ExtractTextFromParagraph)
                        .Where(text => !string.IsNullOrWhiteSpace(text)));
                    cellTexts.Add(cellText);
                }
                textBuilder.AppendLine(string.Join(" | ", cellTexts));
            }

            textBuilder.AppendLine("[/TABLE]");
            return textBuilder.ToString();
        }

        /// <summary>
        /// Extracts headers text
        /// </summary>
        private static string ExtractHeadersText(MainDocumentPart mainPart)
        {
            var textBuilder = new StringBuilder();

            foreach (var headerPart in mainPart.HeaderParts)
            {
                if (headerPart.Header != null)
                {
                    var headerText = ExtractTextFromBody(headerPart.Header);
                    if (!string.IsNullOrWhiteSpace(headerText))
                        textBuilder.AppendLine(headerText);
                }
            }

            return textBuilder.ToString().Trim();
        }

        /// <summary>
        /// Extracts footers text
        /// </summary>
        private static string ExtractFootersText(MainDocumentPart mainPart)
        {
            var textBuilder = new StringBuilder();

            foreach (var footerPart in mainPart.FooterParts)
            {
                if (footerPart.Footer != null)
                {
                    var footerText = ExtractTextFromBody(footerPart.Footer);
                    if (!string.IsNullOrWhiteSpace(footerText))
                        textBuilder.AppendLine(footerText);
                }
            }

            return textBuilder.ToString().Trim();
        }

        /// <summary>
        /// Extracts paragraphs as a list
        /// </summary>
        private static List<string> ExtractParagraphs(Body body)
        {
            return body.Elements<Paragraph>()
                .Select(ExtractTextFromParagraph)
                .Where(text => !string.IsNullOrWhiteSpace(text))
                .ToList();
        }

        /// <summary>
        /// Extracts tables as a list
        /// </summary>
        private static List<string> ExtractTables(Body body)
        {
            return body.Elements<Table>()
                .Select(ExtractTextFromTable)
                .Where(text => !string.IsNullOrWhiteSpace(text))
                .ToList();
        }

        /// <summary>
        /// Extracts headers as a list
        /// </summary>
        private static List<string> ExtractHeadersList(MainDocumentPart mainPart)
        {
            var headers = new List<string>();

            foreach (var headerPart in mainPart.HeaderParts)
            {
                if (headerPart.Header != null)
                {
                    var headerText = ExtractTextFromBody(headerPart.Header);
                    if (!string.IsNullOrWhiteSpace(headerText))
                        headers.Add(headerText);
                }
            }

            return headers;
        }

        /// <summary>
        /// Extracts footers as a list
        /// </summary>
        private static List<string> ExtractFootersList(MainDocumentPart mainPart)
        {
            var footers = new List<string>();

            foreach (var footerPart in mainPart.FooterParts)
            {
                if (footerPart.Footer != null)
                {
                    var footerText = ExtractTextFromBody(footerPart.Footer);
                    if (!string.IsNullOrWhiteSpace(footerText))
                        footers.Add(footerText);
                }
            }

            return footers;
        }

        /// <summary>
        /// Extracts document properties
        /// </summary>
        private static Dictionary<string, string> ExtractDocumentProperties(WordprocessingDocument document)
        {
            var properties = new Dictionary<string, string>();

            try
            {
                var coreProps = document.CoreFilePropertiesPart?.GetXDocument();
                var appProps = document.ExtendedFilePropertiesPart?.GetXDocument();

                if (coreProps != null)
                {
                    var dcNamespace = "http://purl.org/dc/elements/1.1/";
                    var cpNamespace = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties";

                    properties["Title"] = coreProps.Descendants().FirstOrDefault(e => e.Name.LocalName == "title")?.Value ?? "";
                    properties["Subject"] = coreProps.Descendants().FirstOrDefault(e => e.Name.LocalName == "subject")?.Value ?? "";
                    properties["Creator"] = coreProps.Descendants().FirstOrDefault(e => e.Name.LocalName == "creator")?.Value ?? "";
                    properties["Description"] = coreProps.Descendants().FirstOrDefault(e => e.Name.LocalName == "description")?.Value ?? "";
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error extracting document properties: {ex.Message}");
            }

            return properties;
        }

        /// <summary>
        /// Saves extracted text to a file
        /// </summary>
        /// <param name="filePath">Path to the .docx file</param>
        /// <param name="outputPath">Path where to save the text file</param>
        public static void SaveToTextFile(string filePath, string outputPath)
        {
            try
            {
                var extractedText = ExtractAllText(filePath);
                File.WriteAllText(outputPath, extractedText, Encoding.UTF8);
                Console.WriteLine($"Text extracted and saved to: {outputPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error saving to text file: {ex.Message}");
            }
        }

        /// <summary>
        /// Gets word count from the document
        /// </summary>
        /// <param name="filePath">Path to the .docx file</param>
        /// <returns>Approximate word count</returns>
        public static int GetWordCount(string filePath)
        {
            try
            {
                var text = ExtractMainText(filePath);
                return text.Split(new char[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries).Length;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error getting word count: {ex.Message}");
                return 0;
            }
        }
    }

    /// <summary>
    /// Data structure to hold structured document content
    /// </summary>
    public class DocumentContent
    {
        public List<string> Paragraphs { get; set; } = new List<string>();
        public List<string> Tables { get; set; } = new List<string>();
        public List<string> Headers { get; set; } = new List<string>();
        public List<string> Footers { get; set; } = new List<string>();
        public Dictionary<string, string> Properties { get; set; } = new Dictionary<string, string>();

        public override string ToString()
        {
            var sb = new StringBuilder();
            
            if (Properties.Any())
            {
                sb.AppendLine("=== DOCUMENT PROPERTIES ===");
                foreach (var prop in Properties)
                {
                    if (!string.IsNullOrEmpty(prop.Value))
                        sb.AppendLine($"{prop.Key}: {prop.Value}");
                }
                sb.AppendLine();
            }

            if (Headers.Any())
            {
                sb.AppendLine("=== HEADERS ===");
                Headers.ForEach(h => sb.AppendLine(h));
                sb.AppendLine();
            }

            if (Paragraphs.Any())
            {
                sb.AppendLine("=== MAIN CONTENT ===");
                Paragraphs.ForEach(p => sb.AppendLine(p));
                sb.AppendLine();
            }

            if (Tables.Any())
            {
                sb.AppendLine("=== TABLES ===");
                Tables.ForEach(t => sb.AppendLine(t));
                sb.AppendLine();
            }

            if (Footers.Any())
            {
                sb.AppendLine("=== FOOTERS ===");
                Footers.ForEach(f => sb.AppendLine(f));
            }

            return sb.ToString();
        }
    }

    // Example usage class
    public class Program
    {
        public static void Main(string[] args)
        {
            string wordFile = @"C:\path\to\your\document.docx";
            string outputFile = @"C:\path\to\output\extracted_text.txt";

            // Example 1: Extract all text
            Console.WriteLine("=== Extracting all text ===");
            string allText = DocumentTextExtractor.ExtractAllText(wordFile);
            Console.WriteLine(allText);

            // Example 2: Extract main text only
            Console.WriteLine("\n=== Extracting main text only ===");
            string mainText = DocumentTextExtractor.ExtractMainText(wordFile);
            Console.WriteLine(mainText);

            // Example 3: Extract structured content
            Console.WriteLine("\n=== Extracting structured content ===");
            var structuredContent = DocumentTextExtractor.ExtractStructuredContent(wordFile);
            Console.WriteLine(structuredContent.ToString());

            // Example 4: Get word count
            int wordCount = DocumentTextExtractor.GetWordCount(wordFile);
            Console.WriteLine($"\nWord count: {wordCount}");

            // Example 5: Save to text file
            Console.WriteLine("\n=== Saving to text file ===");
            DocumentTextExtractor.SaveToTextFile(wordFile, outputFile);

            // Example 6: Access individual components
            Console.WriteLine("\n=== Accessing individual components ===");
            Console.WriteLine($"Number of paragraphs: {structuredContent.Paragraphs.Count}");
            Console.WriteLine($"Number of tables: {structuredContent.Tables.Count}");
            Console.WriteLine($"Number of headers: {structuredContent.Headers.Count}");
            Console.WriteLine($"Number of footers: {structuredContent.Footers.Count}");

            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
        }
    }
}
