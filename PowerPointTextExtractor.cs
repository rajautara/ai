using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Presentation;
using A = DocumentFormat.OpenXml.Drawing;

namespace PowerPointTextExtractor
{
    public class PowerPointTextExtractor
    {
        /// <summary>
        /// Extracts text from all slides in a PowerPoint presentation
        /// </summary>
        /// <param name="filePath">Path to the .pptx file</param>
        /// <returns>Dictionary with slide numbers and their text content</returns>
        public static Dictionary<int, string> ExtractTextFromAllSlides(string filePath)
        {
            var slideTexts = new Dictionary<int, string>();

            try
            {
                using (var presentationDocument = PresentationDocument.Open(filePath, false))
                {
                    var presentationPart = presentationDocument.PresentationPart;
                    var slideIds = presentationPart.Presentation.SlideIdList.Elements<SlideId>();

                    int slideNumber = 1;
                    foreach (var slideId in slideIds)
                    {
                        var slidePart = (SlidePart)presentationPart.GetPartById(slideId.RelationshipId);
                        var slideText = ExtractTextFromSlide(slidePart);
                        slideTexts.Add(slideNumber, slideText);
                        slideNumber++;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading PowerPoint file: {ex.Message}");
            }

            return slideTexts;
        }

        /// <summary>
        /// Extracts text content from a single slide
        /// </summary>
        /// <param name="slidePart">The slide part to extract text from</param>
        /// <returns>Combined text content from the slide</returns>
        private static string ExtractTextFromSlide(SlidePart slidePart)
        {
            var textBuilder = new StringBuilder();

            // Extract text from slide content
            if (slidePart.Slide != null)
            {
                var textElements = slidePart.Slide.Descendants<A.Text>();
                foreach (var textElement in textElements)
                {
                    if (!string.IsNullOrWhiteSpace(textElement.Text))
                    {
                        textBuilder.AppendLine(textElement.Text.Trim());
                    }
                }
            }

            return textBuilder.ToString().Trim();
        }

        /// <summary>
        /// Extracts text from specific slide by index (1-based)
        /// </summary>
        /// <param name="filePath">Path to the .pptx file</param>
        /// <param name="slideIndex">1-based slide index</param>
        /// <returns>Text content of the specified slide</returns>
        public static string ExtractTextFromSlide(string filePath, int slideIndex)
        {
            try
            {
                using (var presentationDocument = PresentationDocument.Open(filePath, false))
                {
                    var presentationPart = presentationDocument.PresentationPart;
                    var slideIds = presentationPart.Presentation.SlideIdList.Elements<SlideId>().ToList();

                    if (slideIndex < 1 || slideIndex > slideIds.Count)
                    {
                        throw new ArgumentOutOfRangeException(nameof(slideIndex), 
                            $"Slide index must be between 1 and {slideIds.Count}");
                    }

                    var slideId = slideIds[slideIndex - 1];
                    var slidePart = (SlidePart)presentationPart.GetPartById(slideId.RelationshipId);
                    
                    return ExtractTextFromSlide(slidePart);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading slide {slideIndex}: {ex.Message}");
                return string.Empty;
            }
        }

        /// <summary>
        /// Saves extracted text to individual files for each slide
        /// </summary>
        /// <param name="filePath">Path to the .pptx file</param>
        /// <param name="outputDirectory">Directory to save text files</param>
        public static void SaveSlidesToTextFiles(string filePath, string outputDirectory)
        {
            var slideTexts = ExtractTextFromAllSlides(filePath);
            
            if (!Directory.Exists(outputDirectory))
            {
                Directory.CreateDirectory(outputDirectory);
            }

            var fileName = Path.GetFileNameWithoutExtension(filePath);

            foreach (var slide in slideTexts)
            {
                var textFileName = Path.Combine(outputDirectory, $"{fileName}_Slide_{slide.Key:D2}.txt");
                File.WriteAllText(textFileName, slide.Value, Encoding.UTF8);
                Console.WriteLine($"Saved slide {slide.Key} to: {textFileName}");
            }
        }

        /// <summary>
        /// Gets slide count from PowerPoint presentation
        /// </summary>
        /// <param name="filePath">Path to the .pptx file</param>
        /// <returns>Number of slides in the presentation</returns>
        public static int GetSlideCount(string filePath)
        {
            try
            {
                using (var presentationDocument = PresentationDocument.Open(filePath, false))
                {
                    var presentationPart = presentationDocument.PresentationPart;
                    return presentationPart.Presentation.SlideIdList.Elements<SlideId>().Count();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error getting slide count: {ex.Message}");
                return 0;
            }
        }
    }

    // Example usage class
    public class Program
    {
        public static void Main(string[] args)
        {
            string powerPointFile = @"C:\path\to\your\presentation.pptx";
            string outputFolder = @"C:\path\to\output\folder";

            // Example 1: Extract text from all slides
            Console.WriteLine("=== Extracting text from all slides ===");
            var allSlideTexts = PowerPointTextExtractor.ExtractTextFromAllSlides(powerPointFile);
            
            foreach (var slide in allSlideTexts)
            {
                Console.WriteLine($"\n--- Slide {slide.Key} ---");
                Console.WriteLine(slide.Value);
                Console.WriteLine(new string('-', 50));
            }

            // Example 2: Extract text from specific slide
            Console.WriteLine("\n=== Extracting text from slide 1 ===");
            string slide1Text = PowerPointTextExtractor.ExtractTextFromSlide(powerPointFile, 1);
            Console.WriteLine(slide1Text);

            // Example 3: Get slide count
            int slideCount = PowerPointTextExtractor.GetSlideCount(powerPointFile);
            Console.WriteLine($"\nTotal slides in presentation: {slideCount}");

            // Example 4: Save all slides to text files
            Console.WriteLine("\n=== Saving slides to text files ===");
            PowerPointTextExtractor.SaveSlidesToTextFiles(powerPointFile, outputFolder);

            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
        }
    }
}

// Additional helper class for more advanced text extraction
public static class AdvancedTextExtractor
{
    /// <summary>
    /// Extracts text with formatting information preserved
    /// </summary>
    /// <param name="filePath">Path to the .pptx file</param>
    /// <returns>Dictionary with slide info including titles and content separately</returns>
    public static Dictionary<int, SlideInfo> ExtractFormattedText(string filePath)
    {
        var slides = new Dictionary<int, SlideInfo>();

        try
        {
            using (var presentationDocument = PresentationDocument.Open(filePath, false))
            {
                var presentationPart = presentationDocument.PresentationPart;
                var slideIds = presentationPart.Presentation.SlideIdList.Elements<SlideId>();

                int slideNumber = 1;
                foreach (var slideId in slideIds)
                {
                    var slidePart = (SlidePart)presentationPart.GetPartById(slideId.RelationshipId);
                    var slideInfo = ExtractSlideInfo(slidePart, slideNumber);
                    slides.Add(slideNumber, slideInfo);
                    slideNumber++;
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error extracting formatted text: {ex.Message}");
        }

        return slides;
    }

    private static SlideInfo ExtractSlideInfo(SlidePart slidePart, int slideNumber)
    {
        var slideInfo = new SlideInfo { SlideNumber = slideNumber };
        var titleBuilder = new StringBuilder();
        var contentBuilder = new StringBuilder();

        if (slidePart.Slide != null)
        {
            // Extract title (usually first text box or specific shape)
            var shapes = slidePart.Slide.CommonSlideData.ShapeTree.Elements<Shape>();
            
            foreach (var shape in shapes)
            {
                var textBody = shape.TextBody;
                if (textBody != null)
                {
                    var paragraphs = textBody.Elements<A.Paragraph>();
                    var shapeText = string.Join(" ", paragraphs.SelectMany(p => p.Descendants<A.Text>())
                        .Select(t => t.Text?.Trim()).Where(t => !string.IsNullOrEmpty(t)));
                    
                    if (!string.IsNullOrWhiteSpace(shapeText))
                    {
                        // Simple heuristic: first non-empty text is likely the title
                        if (string.IsNullOrEmpty(slideInfo.Title))
                        {
                            slideInfo.Title = shapeText;
                        }
                        else
                        {
                            contentBuilder.AppendLine(shapeText);
                        }
                    }
                }
            }
        }

        slideInfo.Content = contentBuilder.ToString().Trim();
        return slideInfo;
    }
}

// Data structure to hold slide information
public class SlideInfo
{
    public int SlideNumber { get; set; }
    public string Title { get; set; } = string.Empty;
    public string Content { get; set; } = string.Empty;
    
    public override string ToString()
    {
        return $"Slide {SlideNumber}\nTitle: {Title}\nContent: {Content}";
    }
}
