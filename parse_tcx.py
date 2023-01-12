import pandas as pd
from lxml import etree


def extract_tcx_fields(element):
    """Yields (name, value) data points recursively through a TCX XML element"""

    for el in element.iter():
        # Data in a TCX file are not consistent - some are stored as attributes
        for key, value in el.attrib.items():
            localname = etree.QName(key).localname
            if localname != 'type':
                yield (localname, value)

        # Others bury the text one level down in a 'Value' element
        if el.text is None or el.text.isspace():
            child = next(el.iterchildren())
            parent_localname = etree.QName(el).localname
            child_localname = etree.QName(child).localname
            if child_localname == 'Value':
                yield (parent_localname, child.text)

        # But most are recorded as leaf element text
        else:
            localname = etree.QName(el).localname
            if localname != 'Value':
                yield (localname, el.text)


def cleanup_tcx_dataframe(df, time_col):
    """Common post-processing for DataFrames extracted from a TCX XML element"""

    df = df.convert_dtypes().astype(float, errors='ignore')
    df[time_col] = pd.to_datetime(df[time_col])

    # Conversion from meters and m/s to km and kph is done to align with processing done
    # by fitdecode.StandardUnitsDataProcessor
    for col in df.columns:
        if 'Distance' in col:
            df[col] = df[col] / 1000.0
            df = df.rename(columns={col: col.replace('Meters', 'Km')})
        if 'Speed' in col:
            df[col] = df[col] * 60.0 * 60.0 / 1000.0

    return df


def parse_tcx(file):
    """Loads a TCX activity into Pandas DataFrames

    Assumes that the TCX file is all one activity. Files with multiple activities will be
    merged into one set of return values, possibly over-writing some fields.

    Arguments:
        file: File-like or path-like object. A path-like argument ending in .gz will be
              transparently unzipped before processing.

    Returns:
        A tuple of (points, laps, extra)

        points: Time-indexed DataFrame of sensor data recorded during activity
        laps: DataFrame of lap information from the activity
        extra: Dict of selected additional information from the activity
    """

    # Note lxml takes care of identifying and handling a gzipped file
    parser = etree.XMLParser(recover=True)
    root = etree.parse(file, parser).getroot()

    # Note TCX files occasionally have duplicate timestamps, just drop those
    points = pd.DataFrame(dict(extract_tcx_fields(element))
                          for element in root.iter('{*}Trackpoint'))
    points = cleanup_tcx_dataframe(points, 'Time').set_index('Time')
    points = points[~points.index.duplicated()]

    # This drops all Trackpoints from the XML data so they don't show up in lap data
    for element in root.iterfind('.//{*}Track'):
        element.getparent().remove(element)

    laps = pd.DataFrame(dict(extract_tcx_fields(element))
                        for element in root.iter('{*}Lap'))
    laps = cleanup_tcx_dataframe(laps, 'StartTime')

    extra = dict()
    for element in root.iter('{*}Creator'):
        extra.update(dict(extract_tcx_fields(element)))

    return points, laps, extra
